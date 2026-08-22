# src/tools.py
import pandas as pd
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from .config import build_embeddings, CHROMA_COLLECTION, CHROMA_DIR

# ------------------------------------------------------------------
# GLOBAL CONTEXT & DATA LOADING
# ------------------------------------------------------------------

_current_user_id: Optional[int] = None
_current_thread_id: Optional[str] = None


def set_user_id(uid: int):
    global _current_user_id
    _current_user_id = uid


def get_user_id() -> Optional[int]:
    return _current_user_id


def set_thread_id(tid: str):
    global _current_thread_id
    _current_thread_id = tid


# Load DataFrames safely
try:
    # We load these globally so they are available to the tools instantly
    products = pd.read_csv("./dataset/products.csv")
    departments = pd.read_csv("./dataset/departments.csv")
    aisles = pd.read_csv("./dataset/aisles.csv")
    prior = pd.read_csv("./dataset/order_products__prior.csv")
    orders = pd.read_csv("./dataset/orders.csv")

    # Pre-compute lookups for faster access
    _product_lookup = dict(zip(products["product_id"], products["product_name"]))
    DEPARTMENT_NAMES = sorted(departments["department"].dropna().unique().tolist())
    VALID_USER_IDS = sorted(orders["user_id"].dropna().unique().tolist())
    DEFAULT_USER_ID = VALID_USER_IDS[0] if VALID_USER_IDS else 1
except Exception as e:
    print(
        f"WARNING: Could not load datasets. Ensure 'download_dataset.py' has run. Error: {e}"
    )
    DEPARTMENT_NAMES = []
    DEFAULT_USER_ID = 1
    _product_lookup = {}

# ------------------------------------------------------------------
# VECTOR STORE (RAG) SETUP
# ------------------------------------------------------------------
_vector_store = None


def get_vector_store():
    """Singleton accessor for the Chroma Vector Store."""
    global _vector_store
    if _vector_store is None:
        embeddings = build_embeddings()
        _vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return _vector_store


def make_query_prompt(query: str) -> str:
    """Formats the query for better embedding retrieval."""
    return f"Represent this sentence for searching relevant passages: {query.strip().replace(chr(10), ' ')}"


# ------------------------------------------------------------------
# SEARCH TOOLS
# ------------------------------------------------------------------


def search_products(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform a semantic vector search over the product catalog.
    """
    try:
        vector_store = get_vector_store()
        query_prompt = make_query_prompt(query)

        # Perform similarity search
        results = vector_store.similarity_search(query_prompt, k=top_k)

        formatted_results = []
        for doc in results:
            formatted_results.append(
                {
                    "product_id": doc.metadata.get("product_id"),
                    "product_name": doc.metadata.get("product_name"),
                    "aisle": doc.metadata.get("aisle"),
                    "department": doc.metadata.get("department"),
                    "text": doc.page_content,
                }
            )
        return formatted_results
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []


@tool
def search_tool(query: str) -> str:
    """
    Look up products by meaning (e.g., "healthy snacks", "chocolate alternatives").
    Returns a list of matching products with their IDs and locations.
    """
    matches = search_products(query)

    if not matches:
        return "No products found matching your search."

    lines = []
    for item in matches:
        lines.append(
            f"- {item['product_name']} (ID: {item['product_id']})\n"
            f"  Aisle: {item['aisle']}\n"
            f"  Department: {item['department']}\n"
            f"  Details: {item['text'][:100]}..."
        )
    return "\n".join(lines)


@tool
def structured_search_tool(
    product_name: Optional[str] = None,
    department: Optional[str] = None,
    aisle: Optional[str] = None,
    reordered: Optional[bool] = None,
    min_orders: Optional[int] = None,
    order_by: Optional[Literal["count", "add_to_cart_order"]] = None,
    ascending: Optional[bool] = False,
    top_k: Optional[int] = None,
    group_by: Optional[Literal["department", "aisle"]] = None,
    history_only: Optional[bool] = False,
) -> List[Dict[str, Any]]:
    """
    Filter the catalog using SQL-like criteria.
    Use history_only=True to search the user's past purchases.
    """
    try:
        # 1. Base Dataset Selection
        if history_only:
            user_id = get_user_id() or DEFAULT_USER_ID
            # Optimization: Filter orders for user FIRST before merging
            user_orders = orders[orders["user_id"] == user_id]["order_id"]

            if user_orders.empty:
                return []

            # Get products from those orders
            user_products = prior[prior["order_id"].isin(user_orders)]
            # Join with product details
            df = user_products.merge(products, on="product_id")

            # Aggregate history stats
            stats = (
                df.groupby("product_id")
                .agg(
                    count=("product_id", "count"),
                    avg_cart_pos=("add_to_cart_order", "mean"),
                    reordered_sum=("reordered", "sum"),
                )
                .reset_index()
            )

            # Merge stats back to unique product info
            df_unique = products[
                products["product_id"].isin(stats["product_id"])
            ].copy()
            df = df_unique.merge(stats, on="product_id")

        else:
            df = products.copy()

        # Join aisle/dept names for text filtering
        df = df.merge(aisles, on="aisle_id", how="left")
        df = df.merge(departments, on="department_id", how="left")

        # 2. Apply Filters
        if product_name:
            df = df[df["product_name"].str.contains(product_name, case=False, na=False)]

        if department:
            df = df[df["department"] == department]

        if aisle:
            df = df[df["aisle"].str.contains(aisle, case=False, na=False)]

        if history_only:
            if reordered is True:
                df = df[df["reordered_sum"] > 0]
            if min_orders:
                df = df[df["count"] >= min_orders]

        # 3. Grouping
        if group_by:
            if group_by not in df.columns:
                return [{"error": f"Cannot group by {group_by}"}]

            counts = df[group_by].value_counts().reset_index()
            counts.columns = [group_by, "num_products"]
            return counts.to_dict(orient="records")

        # 4. Sorting
        if order_by and history_only:
            col_map = {"count": "count", "add_to_cart_order": "avg_cart_pos"}
            sort_col = col_map.get(order_by)
            if sort_col:
                df = df.sort_values(by=sort_col, ascending=ascending)

        # 5. Limit and Return
        if top_k:
            df = df.head(top_k)

        return df.to_dict(orient="records")

    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


# ------------------------------------------------------------------
# CART TOOLS
# ------------------------------------------------------------------

_cart_storage: Dict[str, Dict[int, int]] = {}


def get_cart() -> Union[List[str], Dict[int, int]]:
    if _current_thread_id is None:
        return ["Session error: no thread ID set."]
    return _cart_storage.setdefault(_current_thread_id, {})


@tool
def cart_tool(
    cart_operation: Literal["add", "remove", "update", "buy"],
    product_id: Optional[int] = None,
    quantity: int = 1,
) -> str:
    """
    Manage the shopping cart.
    Operations: 'add', 'remove', 'update', 'buy'.
    """
    cart = get_cart()
    if isinstance(cart, list):
        return cart[0]  # Error state

    try:
        if cart_operation == "buy":
            if not cart:
                return "Cart is empty."
            cart.clear()
            return "Order processed successfully! Cart cleared."

        if not product_id:
            return "Product ID required."

        product_name = _product_lookup.get(product_id, "Unknown Product")

        if cart_operation == "add":
            cart[product_id] = cart.get(product_id, 0) + quantity
            return f"Added {quantity} x {product_name} (ID: {product_id}). Total: {cart[product_id]}"

        elif cart_operation == "update":
            if product_id not in cart:
                return "Item not in cart."
            cart[product_id] = quantity
            return f"Updated {product_name} quantity to {quantity}."

        elif cart_operation == "remove":
            if product_id not in cart:
                return "Item not in cart."
            current_qty = cart[product_id]
            if quantity >= current_qty:
                del cart[product_id]
                return f"Removed {product_name} from cart."
            else:
                cart[product_id] -= quantity
                return f"Removed {quantity} x {product_name}. Remaining: {cart[product_id]}"

    except Exception as e:
        return f"Cart Error: {str(e)}"

    return "Invalid operation"


@tool
def view_cart() -> str:
    """Returns the current cart contents."""
    cart = get_cart()
    if isinstance(cart, list):
        return cart[0]

    if not cart:
        return "Your cart is empty."

    lines = ["Your cart contains:"]
    for pid, qty in cart.items():
        name = _product_lookup.get(pid, "Unknown Product")
        lines.append(f"- {name} (ID: {pid}) x {qty}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# ESCALATION TOOLS & ERROR HANDLING
# ------------------------------------------------------------------


class RouteToCustomerSupport(BaseModel):
    """Signal to escalate the conversation to human support."""

    reason: str = Field(description="The reason why the customer needs support.")


class EscalateToHuman(BaseModel):
    """Submit a request for human supervisor approval (Support Agent only)."""

    severity: Literal["low", "medium", "high"] = Field(
        description="Severity of the issue."
    )
    summary: str = Field(description="Brief summary of the issue.")


class Search(BaseModel):
    query: str = Field(description="Natural language search query")


def handle_tool_error(state: Dict[str, Any]) -> dict:
    """Fallback function for when tools fail."""
    error = state.get("error")
    try:
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\nPlease fix your arguments and try again.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }
    except:
        return {"messages": []}


def create_tool_node_with_fallback(tools: list) -> ToolNode:
    """
    Create a ToolNode that handles errors automatically.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


__all__ = [
    "RouteToCustomerSupport",
    "EscalateToHuman",
    "Search",
    "search_products",
    "search_tool",
    "cart_tool",
    "view_cart",
    "handle_tool_error",
    "create_tool_node_with_fallback",
    "structured_search_tool",
]
