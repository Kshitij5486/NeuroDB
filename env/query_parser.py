"""
query_parser.py
NeuroDB - Converts SQL query into feature vector for RL agent
"""

import numpy as np
import pglast
from pglast import parse_sql
from pglast.visitors import Visitor

TPCH_TABLES = [
    "lineitem", "orders", "customer", "part",
    "supplier", "partsupp", "nation", "region"
]
TABLE_INDEX  = {t: i for i, t in enumerate(TPCH_TABLES)}
NUM_TABLES   = len(TPCH_TABLES)
FEATURE_DIM  = 1 + NUM_TABLES + 1 + 1 + 1 + 1  # 13


class QueryVisitor(Visitor):

    def __init__(self):
        self.tables     : list[str] = []
        self.predicates : int       = 0
        self.columns    : int       = 0
        self.has_groupby: bool      = False
        self.has_orderby: bool      = False
        self._alias_map : dict      = {}

    def visit_RangeVar(self, ancestors, node):
        name = node.relname.lower()
        if node.alias:
            self._alias_map[node.alias.aliasname.lower()] = name
        if name not in self.tables:
            self.tables.append(name)

    def visit_ResTarget(self, ancestors, node):
        self.columns += 1

    def visit_A_Expr(self, ancestors, node):
        self.predicates += 1

    def visit_SortBy(self, ancestors, node):
        self.has_orderby = True


def extract_features(sql: str) -> tuple[np.ndarray, list[str]]:
    try:
        ast = parse_sql(sql)
    except pglast.Error as e:
        raise ValueError(f"SQL parse error: {e}") from e

    visitor = QueryVisitor()
    visitor(ast)
    tables = visitor.tables

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    vec[0] = min(len(tables), 6) / 6.0

    for t in tables:
        idx = TABLE_INDEX.get(t)
        if idx is not None:
            vec[1 + idx] = 1.0

    vec[1 + NUM_TABLES]     = min(visitor.predicates, 20) / 20.0
    vec[1 + NUM_TABLES + 1] = min(visitor.columns,    30) / 30.0
    vec[1 + NUM_TABLES + 2] = 1.0 if visitor.has_groupby else 0.0
    vec[1 + NUM_TABLES + 3] = 1.0 if visitor.has_orderby else 0.0

    return vec, tables


if __name__ == "__main__":
    queries = [
        """SELECT c.c_name, SUM(o.o_totalprice)
           FROM customer c
           JOIN orders o ON c.c_custkey = o.o_custkey
           WHERE o.o_orderdate >= '1995-01-01'
           GROUP BY c.c_name ORDER BY 1""",

        """SELECT l.l_orderkey, o.o_orderdate, c.c_name
           FROM lineitem l
           JOIN orders o ON l.l_orderkey = o.o_orderkey
           JOIN customer c ON o.o_custkey = c.c_custkey
           WHERE l.l_shipdate < '1995-03-15'
           ORDER BY l.l_orderkey""",

        """SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount))
           FROM customer c
           JOIN orders o   ON c.c_custkey  = o.o_custkey
           JOIN lineitem l ON l.l_orderkey = o.o_orderkey
           JOIN supplier s ON l.l_suppkey  = s.s_suppkey
           JOIN nation n   ON s.s_nationkey= n.n_nationkey
           WHERE n.n_name = 'GERMANY'
           GROUP BY n.n_name ORDER BY 2 DESC""",
    ]

    print(f"Feature dimension: {FEATURE_DIM}\n" + "-" * 50)
    for i, sql in enumerate(queries, 1):
        vec, tables = extract_features(sql)
        print(f"\nQuery {i}  tables={tables}")
        print(f"  vec = {np.round(vec, 2)}")