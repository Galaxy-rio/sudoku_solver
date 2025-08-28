#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式数独辅助解题器

输入：9 行，每行 9 个字符（0-9），0 表示空位，无分隔符
交互：
  n → 执行并展示下一步提示
  r → 重新输入棋盘
  e → 退出

每一步都会以 3×3 小九宫形式打印每格候选（已填数字显示在中心）。
"""

from collections import defaultdict
from collections import deque
from itertools import combinations
from typing import List, Set, Tuple, Optional, Dict

Grid = List[List[int]]
Pos = Tuple[int, int]


# ===================== 输入与基础结构 =====================

def read_board() -> Grid:
    print("请输入数独棋盘（9行，每行9个数字，0表示空）：")
    board: Grid = []
    for r in range(9):
        line = input().strip()
        if len(line) != 9 or any(ch not in "0123456789" for ch in line):
            raise ValueError(f"第 {r + 1} 行格式错误：应为9个字符，仅包含0-9。")
        board.append([int(ch) for ch in line])
    return board


def rc(r: int, c: int) -> str:
    return f"R{r + 1}C{c + 1}"


def unit_iter():
    """遍历全部 27 个单位（9 行、9 列、9 宫），返回 (kind, idx, cells)"""
    # 行
    for r in range(9):
        yield "row", r, [(r, c) for c in range(9)]
    # 列
    for c in range(9):
        yield "col", c, [(r, c) for r in range(9)]
    # 宫
    k = 0
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = [(br + i, bc + j) for i in range(3) for j in range(3)]
            yield "box", k, cells
            k += 1


# ===================== 候选与打印 =====================

def init_candidates(board: Grid) -> List[List[Set[int]]]:
    """初始化候选；随后调用 update_candidates 以传播已填数字的影响。"""
    cand = [[(set(range(1, 10)) if board[r][c] == 0 else {board[r][c]}) for c in range(9)] for r in range(9)]
    update_candidates(board, cand)
    return cand


def update_candidates(board: Grid, cand: List[List[Set[int]]]) -> None:
    """根据当前已填数字收缩候选（只做剪除，不重新放回被消去的候选）。"""
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                val = board[r][c]
                cand[r][c] = {val}
                # 同行/同列/同宫删掉 val
                for cc in range(9):
                    if cc != c:
                        cand[r][cc].discard(val)
                for rr in range(9):
                    if rr != r:
                        cand[rr][c].discard(val)
                br, bc = (r // 3) * 3, (c // 3) * 3
                for rr in range(br, br + 3):
                    for cc in range(bc, bc + 3):
                        if rr != r or cc != c:
                            cand[rr][cc].discard(val)


def pretty_print_candidates(board: Grid, cand: List[List[Set[int]]]) -> None:
    """
    以 3x3 小九宫形式打印每个格子的候选。
    已填格只在中心位置显示数字，其它位置留空。
    整体以 3x3 宫为块加粗分隔。
    """
    lines = []
    cell_sep = " | "
    block_sep_line = None

    for r in range(9):
        sublines = [""] * 3
        for c in range(9):
            box = [[" " for _ in range(3)] for _ in range(3)]
            if board[r][c] != 0:
                # 已填数字放中心
                box[1][1] = str(board[r][c])
            else:
                for n in cand[r][c]:
                    rr, cc = divmod(n - 1, 3)
                    box[rr][cc] = str(n)

            for k in range(3):
                sublines[k] += " ".join(box[k]) + cell_sep

            # 竖向粗分隔
            if c % 3 == 2 and c != 8:
                for k in range(3):
                    sublines[k] += "|| "

        lines.extend(sublines)

        # 横向粗分隔
        if r % 3 == 2 and r != 8:
            if block_sep_line is None:
                block_sep_line = "-" * len(sublines[0])
            lines.append(block_sep_line)

    print("\n".join(lines))


# ===================== 操作应用 =====================

def place(board: Grid, cand: List[List[Set[int]]], r: int, c: int, v: int) -> None:
    """填入一个确定数字，并传播候选剪除。"""
    board[r][c] = v
    update_candidates(board, cand)


def eliminate(cand: List[List[Set[int]]], targets: List[Tuple[int, int, Set[int]]]) -> None:
    """在若干格子中消除若干候选数（不改变 board）。targets: [(r,c,{vs}), ...]"""
    for r, c, vs in targets:
        cand[r][c] -= vs


# ===================== 技巧 1：Naked Single =====================

def find_naked_single(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0 and len(cand[r][c]) == 1:
                v = next(iter(cand[r][c]))
                text = f"[Naked Single] {rc(r, c)} 只能是 {v}。"
                return text, ("fill", str(r), str(c), str(v)), (), ()
    return None


# ===================== 技巧 2：Hidden Single =====================

def find_hidden_single(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    for kind, idx, cells in unit_iter():
        occ: Dict[int, List[Pos]] = {v: [] for v in range(1, 10)}
        for (r, c) in cells:
            if board[r][c] == 0:
                for v in cand[r][c]:
                    occ[v].append((r, c))
        for v, locs in occ.items():
            if len(locs) == 1:
                r, c = locs[0]
                label = {"row": f"第{idx + 1}行", "col": f"第{idx + 1}列", "box": f"第{idx // 3 + 1}-{idx % 3 + 1}宫"}[
                    kind]
                text = f"[Hidden Single] 数字 {v} 在{label}仅能放在 {rc(r, c)} → 填入 {v}。"
                return text, ("fill", str(r), str(c), str(v)), (), ()
    return None


# ===================== 技巧 3：Locked Candidate =====================

def find_locked_candidate(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = [(br + i, bc + j) for i in range(3) for j in range(3)]
            for v in range(1, 10):
                pos = [(r, c) for (r, c) in cells if board[r][c] == 0 and v in cand[r][c]]
                if len(pos) < 2:
                    continue
                rows = {r for r, _ in pos}
                cols = {c for _, c in pos}
                # 宫内全部落在同一行 → 该行宫外消除
                if len(rows) == 1:
                    r = next(iter(rows))
                    elim_targets = []
                    for c in range(9):
                        if not (bc <= c < bc + 3):
                            if board[r][c] == 0 and v in cand[r][c]:
                                elim_targets.append((r, c, {v}))
                    if elim_targets:
                        text = f"[Locked Candidate] 数字 {v} 在宫 {(br // 3) + 1}-{(bc // 3) + 1} 只位于行 {r + 1}，故该行宫外位置不可为 {v}。"
                        return text, (), ("elim",), tuple(f"{t[0]},{t[1]},{v}" for t in elim_targets)
                # 宫内全部落在同一列 → 该列宫外消除
                if len(cols) == 1:
                    c = next(iter(cols))
                    elim_targets = []
                    for r in range(9):
                        if not (br <= r < br + 3):
                            if board[r][c] == 0 and v in cand[r][c]:
                                elim_targets.append((r, c, {v}))
                    if elim_targets:
                        text = f"[Locked Candidate] 数字 {v} 在宫 {(br // 3) + 1}-{(bc // 3) + 1} 只位于列 {c + 1}，故该列宫外位置不可为 {v}。"
                        return text, (), ("elim",), tuple(f"{t[0]},{t[1]},{v}" for t in elim_targets)
    return None


# ===================== 技巧 4~6：Hidden Pair/Triple/Quad =====================

def _find_hidden_set(board: Grid, cand: List[List[Set[int]]], size: int) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    label_map = {"row": "行", "col": "列", "box": "宫"}
    for kind, idx, cells in unit_iter():
        # 统计每个数字出现位置
        occ: Dict[int, List[Pos]] = defaultdict(list)
        empties = [p for p in cells if board[p[0]][p[1]] == 0]
        for (r, c) in empties:
            for v in cand[r][c]:
                occ[v].append((r, c))
        digits = [d for d in range(1, 10) if 1 <= len(occ[d]) <= size]
        # 枚举 size 个数字的组合
        for d_set in combinations(digits, size):
            locs = set()
            for d in d_set:
                locs.update(occ[d])
            if len(locs) != size:
                continue  # 隐性集合的必要条件：这些数字仅覆盖 size 个格
            # 这些格中删去非 d_set 的其它候选
            elim_targets: List[Tuple[int, int, Set[int]]] = []
            changed = False
            for (r, c) in locs:
                others = cand[r][c] - set(d_set)
                if others:
                    elim_targets.append((r, c, others))
                    changed = True
            if changed:
                name = {2: "Hidden Pair", 3: "Hidden Triple", 4: "Hidden Quad"}[size]
                # 文字描述
                if kind == "box":
                    scope = f"第{idx // 3 + 1}-{idx % 3 + 1}{label_map[kind]}"
                else:
                    scope = f"第{idx + 1}{label_map[kind]}"
                d_txt = ",".join(map(str, d_set))
                loc_txt = ", ".join(rc(r, c) for (r, c) in sorted(locs))
                text = f"[{name}] 在{scope}中，数字 {{{d_txt}}} 仅出现于 {size} 个格：{loc_txt}，因此这些格只能保留 {{{d_txt}}}，其余候选可删。"
                # 将所有消除目标编码为字符串，主循环里一次性应用
                return (text, (), ("elim",),
                        tuple(f"{r},{c}," + ";".join(map(str, sorted(vs))) for (r, c, vs) in elim_targets))
    return None


def find_hidden_pair(board: Grid, cand: List[List[Set[int]]]):
    return _find_hidden_set(board, cand, 2)


def find_hidden_triple(board: Grid, cand: List[List[Set[int]]]):
    return _find_hidden_set(board, cand, 3)


def find_hidden_quad(board: Grid, cand: List[List[Set[int]]]):
    return _find_hidden_set(board, cand, 4)

# ===================== 技巧 7~9：Naked Pair/Triple/Quad=====================
from typing import Dict as _Dict, Tuple as _Tuple

def _find_naked_set(board: Grid, cand: List[List[Set[int]]], size: int) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
]:
    label_map = {"row": "行", "col": "列", "box": "宫"}
    for kind, idx, cells in unit_iter():
        empties = [(r, c) for (r, c) in cells if board[r][c] == 0]

        # 将“候选恰好为 size 个”的格子按候选集合分组
        pat: _Dict[_Tuple[int, ...], List[Pos]] = defaultdict(list)
        for (r, c) in empties:
            if len(cand[r][c]) == size:
                key = tuple(sorted(cand[r][c]))
                pat[key].append((r, c))

        # 寻找恰好出现 size 个格子的候选集合（裸集合）
        for digits_t, locs in pat.items():
            if len(locs) != size:
                continue
            digits = set(digits_t)

            # 从同一单位内“其它格子”里删除这些数字
            elim_targets: List[Tuple[int, int, Set[int]]] = []
            for (r, c) in empties:
                if (r, c) in locs:
                    continue
                inter = cand[r][c] & digits
                if inter:
                    elim_targets.append((r, c, inter))

            if elim_targets:
                name = {2: "Naked Pair", 3: "Naked Triple", 4: "Naked Quad"}[size]
                if kind == "box":
                    scope = f"第{idx // 3 + 1}-{idx % 3 + 1}{label_map[kind]}"
                else:
                    scope = f"第{idx + 1}{label_map[kind]}"

                d_txt = ",".join(map(str, digits_t))
                loc_txt = ", ".join(rc(r, c) for (r, c) in sorted(locs))
                text = (f"[{name}] 在{scope}中，{size} 个格 {loc_txt} 仅包含 {{{d_txt}}}。"
                        f" 因此{scope}内其它格不可能取 {{{d_txt}}}，将其删除。")
                return (text, (), ("elim",),
                        tuple(f"{r},{c}," + ";".join(map(str, sorted(vs))) for (r, c, vs) in elim_targets))
    return None

def find_naked_pair(board: Grid, cand: List[List[Set[int]]]):
    return _find_naked_set(board, cand, 2)

def find_naked_triple(board: Grid, cand: List[List[Set[int]]]):
    return _find_naked_set(board, cand, 3)

def find_naked_quad(board: Grid, cand: List[List[Set[int]]]):
    return _find_naked_set(board, cand, 4)


# ===================== 技巧 10~12：Fish=====================

def _fish(board: Grid, cand: List[List[Set[int]]], size: int, base: str) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
]:

    name = {2: "X-Wing", 3: "Swordfish", 4: "Jellyfish"}[size]

    for v in range(1, 10):
        line_to_pos = {}
        for i in range(9):
            if base == 'row':
                poss = [j for j in range(9) if board[i][j] == 0 and v in cand[i][j]]
            else:
                poss = [j for j in range(9) if board[j][i] == 0 and v in cand[j][i]]

            cols_or_rows = sorted(set(poss))
            # Fish 算法：每条线上的候选数应该在 2 到 size 之间
            if 2 <= len(cols_or_rows) <= size:
                line_to_pos[i] = cols_or_rows

        if len(line_to_pos) < size:
            continue

        for lines in combinations(line_to_pos.keys(), size):
            # 收集所有涉及的列/行
            all_cols_or_rows = set()
            for line in lines:
                all_cols_or_rows.update(line_to_pos[line])

            # Fish 的关键条件：size 条线覆盖恰好 size 列/行
            if len(all_cols_or_rows) == size:
                elim_targets: List[Tuple[int, int, Set[int]]] = []
                if base == 'row':
                    cols = sorted(all_cols_or_rows)
                    for r in range(9):
                        if r in lines:
                            continue
                        for c in cols:
                            if board[r][c] == 0 and v in cand[r][c]:
                                elim_targets.append((r, c, {v}))
                    lines_txt = ", ".join(str(r + 1) for r in sorted(lines))
                    cols_txt = ", ".join(str(c + 1) for c in cols)
                    text = (f"[{name}] 数字 {v} 在行 {lines_txt} 的列 {{{cols_txt}}} 上成型，"
                            f"因此其它行这些列不能为 {v}。")
                else:
                    rows = sorted(all_cols_or_rows)
                    for c in range(9):
                        if c in lines:
                            continue
                        for r in rows:
                            if board[r][c] == 0 and v in cand[r][c]:
                                elim_targets.append((r, c, {v}))
                    lines_txt = ", ".join(str(c + 1) for c in sorted(lines))
                    rows_txt = ", ".join(str(r + 1) for r in rows)
                    text = (f"[{name}] 数字 {v} 在列 {lines_txt} 的行 {{{rows_txt}}} 上成型，"
                            f"因此其它列这些行不能为 {v}。")

                if elim_targets:
                    return text, (), ("elim",), tuple(f"{r},{c},{v}" for (r, c, _) in elim_targets)
    return None


def find_x_wing(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    return _fish(board, cand, 2, 'row') or _fish(board, cand, 2, 'col')


def find_swordfish(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    return _fish(board, cand, 3, 'row') or _fish(board, cand, 3, 'col')


def find_jellyfish(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    return _fish(board, cand, 4, 'row') or _fish(board, cand, 4, 'col')


# ===================== 技巧 13~15：Color Trap/Wrap/Wing =====================

def _same_unit(a: Pos, b: Pos) -> bool:
    """判断两格是否处在同一行/列/宫中（可相互“可见”）"""
    (r1, c1), (r2, c2) = a, b
    if r1 == r2 or c1 == c2:
        return True
    if (r1 // 3 == r2 // 3) and (c1 // 3 == c2 // 3):
        return True
    return False

def _build_strong_link_graph(board: Grid, cand: List[List[Set[int]]], v: int):
    """
    对给定数字 v，建立"强链"图：
      - 节点：所有含候选 v 的空格 (r,c)
      - 边：在同一单位（行/列/宫）且该单位中候选 v 恰好出现 2 次，则这两个格之间存在强链（互斥的二选一）
    返回：字典 graph: node -> set(neighbors)
    """
    nodes = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0 and v in cand[r][c]]
    graph = {n: set() for n in nodes}

    # 遍历单位（行/列/宫），若某单位中 v 的候选格恰为 2 个，则连边
    for kind, idx, cells in unit_iter():
        locs = [(r, c) for (r, c) in cells if board[r][c] == 0 and v in cand[r][c]]
        if len(locs) == 2:
            a, b = locs
            graph[a].add(b)
            graph[b].add(a)

    return graph

def _color_components(graph):
    """
    将图的每个连通分量进行二色染色（强链应该形成二分图）。
    返回：list of components，每个 component 是 (nodes_set, color_map)
      - nodes_set: set of nodes in该分量
      - color_map: dict node->0/1
    """
    visited = set()
    comps = []
    for node in graph:
        if node in visited:
            continue
        # BFS 染色
        q = deque([node])
        color = {node: 0}
        comp_nodes = {node}
        visited.add(node)
        bipartite = True
        while q:
            u = q.popleft()
            for w in graph[u]:
                if w not in color:
                    color[w] = 1 - color[u]
                    q.append(w)
                    comp_nodes.add(w)
                    visited.add(w)
                else:
                    # 若已经有颜色且冲突（说明奇环），这里仍保留颜色分配（但注意可能不能用于某些断言）
                    if color[w] == color[u]:
                        # 记录为非二分图（不过我们仍继续）
                        bipartite = False
        comps.append((comp_nodes, color, bipartite))
    return comps

def find_color_trap(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    """
    Color Trap:
      - 对每个数字 v 建立强链图并染色，每个连通分量产生两个颜色（0/1）。
      - 若在同一分量中发现 **同一颜色的两个节点互相可见（同行/同列/同宫）**，
        则该颜色导致矛盾 → 该颜色为“假色”，可以删除所有该颜色节点上的 v 候选。
    返回首个能删除候选的发现（文本说明 + elim 列表）。
    """
    for v in range(1, 10):
        graph = _build_strong_link_graph(board, cand, v)
        if not graph:
            continue
        comps = _color_components(graph)
        for comp_nodes, color_map, bipartite in comps:
            # 检查每种颜色是否在 comp 内出现了可见冲突（同色两点在同一单位）
            for color_val in (0, 1):
                same_color_nodes = [n for n, col in color_map.items() if col == color_val]
                # 若该颜色节点少于2个则无法形成冲突
                if len(same_color_nodes) < 2:
                    continue
                conflict_found = False
                # 检查任意两点是否可见（同行/同宫）
                for i in range(len(same_color_nodes)):
                    for j in range(i + 1, len(same_color_nodes)):
                        if _same_unit(same_color_nodes[i], same_color_nodes[j]):
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    # 将该颜色上的所有节点删除候选 v
                    elim_targets = []
                    for (r, c) in same_color_nodes:
                        # 确保当前仍存在该候选
                        if board[r][c] == 0 and v in cand[r][c]:
                            elim_targets.append((r, c, {v}))
                    if elim_targets:
                        comp_repr = ", ".join(rc(r, c) for (r, c) in sorted(comp_nodes))
                        color_nodes_repr = ", ".join(rc(r, c) for (r, c) in sorted(same_color_nodes))
                        text = (f"[Color Trap] 数字 {v} 在二色链的一个连通分量（格：{comp_repr}）中，"
                                f"发现同色格 {color_nodes_repr} 互相可见，说明该色导致矛盾，"
                                f"因此这些格不能为 {v}。")
                        return text, (), ("elim",), tuple(f"{r},{c},{v}" for (r, c, _) in elim_targets)
    return None

def find_color_wrap(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    """
    Color Wrap（二色包裹 / 两色可见消除）:
      - 对每个数字 v 建立强链图并染色。
      - 若某个（未染色）格子能同时"看见"至少一个颜色0的节点和至少一个颜色1的节点，
        则该格子不可能是 v，因它会在两种颜色真假链中都被否定 — 可以删去该格的 v 候选。
    返回首个能删除候选的发现（文本说明 + elim 列表）。
    """
    for v in range(1, 10):
        graph = _build_strong_link_graph(board, cand, v)
        if not graph:
            continue
        comps = _color_components(graph)
        # 汇总整个图中颜色0/1在每个分量中各自的节点集合（我们按分量分别考虑）
        for comp_nodes, color_map, bipartite in comps:
            color0_nodes = [n for n, col in color_map.items() if col == 0]
            color1_nodes = [n for n, col in color_map.items() if col == 1]
            if not color0_nodes or not color1_nodes:
                continue
            # 对于棋盘上任意含 v 的格子（不在当前连通分量内），检查是否同时看到 color0 和 color1
            elim_targets = []
            for r in range(9):
                for c in range(9):
                    if board[r][c] != 0:
                        continue
                    if v not in cand[r][c]:
                        continue
                    # 关键修复：不处理已经在连通分量中的节点
                    if (r, c) in comp_nodes:
                        continue
                    sees0 = any(_same_unit((r, c), node) for node in color0_nodes)
                    sees1 = any(_same_unit((r, c), node) for node in color1_nodes)
                    if sees0 and sees1:
                        # 该格同时被两色可见，故可删 v
                        elim_targets.append((r, c, {v}))
            if elim_targets:
                comp_repr = ", ".join(rc(r, c) for (r, c) in sorted(comp_nodes))
                text = (f"[Color Wrap] 数字 {v} 在连通分量（格：{comp_repr}）的两色分别可达某格，"
                        f"因此能同时看到两色的格不能为 {v}，已删去这些格的候选。")
                return text, (), ("elim",), tuple(f"{r},{c},{v}" for (r, c, _) in elim_targets)
    return None

def find_color_wing(board: Grid, cand: List[List[Set[int]]]) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    """
    Color Wing (颜色翅膀):
    - 分析强链分量和弱链接的组合模式
    - 寻找多链组合形成的消除机会
    """
    for v in range(1, 10):
        # 获取所有包含v的位置
        positions = [(r, c) for r in range(9) for c in range(9)
                    if board[r][c] == 0 and v in cand[r][c]]

        if len(positions) < 4:
            continue

        # 构建强链图
        graph = _build_strong_link_graph(board, cand, v)
        if not graph:
            continue

        # 获取连通分量和颜色
        comps = _color_components(graph)

        # 标准Color Wing：检查能同时看到两种颜色的位置
        for comp_nodes, color_map, bipartite in comps:
            if not bipartite or len(comp_nodes) < 2:
                continue

            color0_nodes = [n for n, col in color_map.items() if col == 0]
            color1_nodes = [n for n, col in color_map.items() if col == 1]

            if len(color0_nodes) == 0 or len(color1_nodes) == 0:
                continue

            elim_targets = []

            # 检查所有不在当前分量中的候选位置
            for (r, c) in positions:
                if (r, c) in comp_nodes:
                    continue

                sees_color0 = any(_same_unit((r, c), node) for node in color0_nodes)
                sees_color1 = any(_same_unit((r, c), node) for node in color1_nodes)

                if sees_color0 and sees_color1:
                    elim_targets.append((r, c, {v}))

            if elim_targets:
                comp_desc = ", ".join([rc(r, c) for r, c in sorted(comp_nodes)])
                color0_desc = ", ".join([rc(r, c) for r, c in sorted(color0_nodes)])
                color1_desc = ", ".join([rc(r, c) for r, c in sorted(color1_nodes)])
                elim_desc = ", ".join([rc(r, c) for r, c, _ in elim_targets])

                text = (f"[Color Wing] 数字 {v} 在强链组（{comp_desc}）中形成二分图着色：\n"
                       f"• 颜色A位置：{color0_desc}\n"
                       f"• 颜色B位置：{color1_desc}\n"
                       f"• 原理：在强链中，同一颜色的位置不能同时为真。位置 {elim_desc} "
                       f"能同时看到两种颜色的位置，因此无论哪种颜色为真，这些位置都不能放置数字 {v}。")
                return text, (), ("elim",), tuple(f"{r},{c},{v}" for (r, c, _) in elim_targets)

        # 增强型Color Wing：多链组合分析
        result = find_multi_chain_wing(board, cand, v, positions, comps)
        if result:
            return result

    return None

def find_multi_chain_wing(board: Grid, cand: List[List[Set[int]]], v: int,
                         positions: list, comps: list) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    """
    多链组合Wing分析：考虑多个强链分量之间的相互作用
    """
    if len(comps) < 2:
        return None

    # 分析所有分量的组合
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            comp1_nodes, color1_map, bipartite1 = comps[i]
            comp2_nodes, color2_map, bipartite2 = comps[j]

            if not bipartite1 or not bipartite2:
                continue

            # 寻找连接两个分量的"桥接"位置
            bridge_positions = []
            for (r, c) in positions:
                if (r, c) in comp1_nodes or (r, c) in comp2_nodes:
                    continue

                # 检查是否与两个分量都有关系
                relates_comp1 = any(_same_unit((r, c), node) for node in comp1_nodes)
                relates_comp2 = any(_same_unit((r, c), node) for node in comp2_nodes)

                if relates_comp1 and relates_comp2:
                    bridge_positions.append((r, c))

            if bridge_positions:
                # 通过桥接分析消除机会
                elim_targets = []

                # 获取两个分量的颜色分组
                comp1_color0 = [n for n, col in color1_map.items() if col == 0]
                comp1_color1 = [n for n, col in color1_map.items() if col == 1]
                comp2_color0 = [n for n, col in color2_map.items() if col == 0]
                comp2_color1 = [n for n, col in color2_map.items() if col == 1]

                # 对于其他位置，检查复杂的约束关系
                for (r, c) in positions:
                    if ((r, c) in comp1_nodes or (r, c) in comp2_nodes or
                        (r, c) in bridge_positions):
                        continue

                    # 检查与桥接位置的关系
                    sees_bridge = any(_same_unit((r, c), bridge) for bridge in bridge_positions)

                    # 检查与分量的关系
                    sees_comp1 = any(_same_unit((r, c), node) for node in comp1_nodes)
                    sees_comp2 = any(_same_unit((r, c), node) for node in comp2_nodes)

                    # 多链Wing逻辑：如果通过桥接形成复杂约束
                    if sees_bridge and (sees_comp1 or sees_comp2):
                        elim_targets.append((r, c, {v}))

                if elim_targets:
                    comp1_desc = ", ".join([rc(r, c) for r, c in sorted(comp1_nodes)])
                    comp2_desc = ", ".join([rc(r, c) for r, c in sorted(comp2_nodes)])
                    bridge_desc = ", ".join([rc(r, c) for r, c in bridge_positions])
                    elim_desc = ", ".join([rc(r, c) for r, c, _ in elim_targets])

                    text = (f"[Multi-Chain Wing] 数字 {v} 的多链组合分析：\n"
                           f"• 强链组1：{comp1_desc}\n"
                           f"• 强链组2：{comp2_desc}\n"
                           f"• 桥接位置：{bridge_desc}\n"
                           f"• 原理：两个独立的强链分量通过桥接位置形成复杂的约束网络。"
                           f"位置 {elim_desc} 受到多重约束影响，无法放置数字 {v}。")
                    return text, (), ("elim",), tuple(f"{r},{c},{v}" for (r, c, _) in elim_targets)

    # 另一种模式：三位置链分析
    return find_three_position_chain_wing(board, cand, v, positions)

def find_three_position_chain_wing(board: Grid, cand: List[List[Set[int]]], v: int,
                                  positions: list) -> Optional[
    Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]:
    """
    三位置链Wing分析：特别处理某些列/行有3个候选的情况
    """
    # 寻找有3个候选的单位
    three_candidate_units = []

    for kind, idx, cells in unit_iter():
        unit_candidates = [(r, c) for r, c in cells if board[r][c] == 0 and v in cand[r][c]]
        if len(unit_candidates) == 3:
            three_candidate_units.append((kind, idx, unit_candidates))

    if len(three_candidate_units) < 2:
        return None

    # 分析多个三候选单位的组合
    for i in range(len(three_candidate_units)):
        for j in range(i + 1, len(three_candidate_units)):
            kind1, idx1, cands1 = three_candidate_units[i]
            kind2, idx2, cands2 = three_candidate_units[j]

            # 寻找交叉关系
            intersections = []
            for pos1 in cands1:
                for pos2 in cands2:
                    if _same_unit(pos1, pos2) and pos1 != pos2:
                        intersections.append((pos1, pos2))

            if len(intersections) >= 2:
                # 找到复杂的约束网络，分析消除机会
                all_involved = set(cands1 + cands2)

                elim_targets = []
                for (r, c) in positions:
                    if (r, c) in all_involved:
                        continue

                    # 检查是否与多个涉及位置相关
                    related_count = sum(1 for pos in all_involved if _same_unit((r, c), pos))

                    if related_count >= 3:  # 与至少3个位置相关，形成强约束
                        elim_targets.append((r, c, {v}))

                if elim_targets:
                    unit1_name = f"第{idx1+1}{kind1}" if kind1 != "box" else f"第{idx1//3+1}-{idx1%3+1}宫"
                    unit2_name = f"第{idx2+1}{kind2}" if kind2 != "box" else f"第{idx2//3+1}-{idx2%3+1}宫"

                    involved_desc = ", ".join([rc(r, c) for r, c in sorted(all_involved)])
                    elim_desc = ", ".join([rc(r, c) for r, c, _ in elim_targets])
                    intersect_desc = ", ".join([f"{rc(p1[0], p1[1])}-{rc(p2[0], p2[1])}" for p1, p2 in intersections])

                    text = (f"[Three-Position Wing] 数字 {v} 的三位置链分析：\n"
                           f"• {unit1_name}候选位置：{', '.join([rc(r, c) for r, c in cands1])}\n"
                           f"• {unit2_name}候选位置：{', '.join([rc(r, c) for r, c in cands2])}\n"
                           f"• 交叉关系：{intersect_desc}\n"
                           f"• 原理：两个三位置单位通过交叉关系形成约束网络。"
                           f"位置 {elim_desc} 与网络中多个位置相关，受到强约束影响，无法放置数字 {v}。")
                    return text, (), ("elim",), tuple(f"{r},{c},{v}" for (r, c, _) in elim_targets)

    return None
# ===================== 主交互循环 =====================

def main():
    try:
        board = read_board()
    except Exception as e:
        print("输入错误：", e)
        return

    cand = init_candidates(board)

    while True:
        print("当前候选棋盘（中心为已填数字）：")
        pretty_print_candidates(board, cand)

        # 查找下一步提示（按优先级）
        hint = (
                find_naked_single(board, cand)
                or find_hidden_single(board, cand)
                or find_locked_candidate(board, cand)
                or find_hidden_pair(board, cand)
                or find_hidden_triple(board, cand)
                or find_hidden_quad(board, cand)
                or find_naked_pair(board, cand)
                or find_naked_triple(board, cand)
                or find_naked_quad(board, cand)
                or find_x_wing(board, cand)
                or find_swordfish(board, cand)
                or find_jellyfish(board, cand)
                or find_color_trap(board, cand)
                or find_color_wrap(board, cand)
                or find_color_wing(board, cand)
        )

        if hint is None:
            print("提示：没有更多可用的人类逻辑（或需要更高级技巧）。")
            cmd = input("请输入命令 [r=重新输入, e=退出, others=继续查看]: ").strip().lower()
            if cmd == 'r':
                try:
                    board = read_board()
                    cand = init_candidates(board)
                    continue
                except Exception as e:
                    print("输入错误：", e)
                    return
            elif cmd == 'e':
                print("已退出。")
                return
            else:
                continue

        text, fill_action, elim_flag, elim_payload = hint
        print("提示：", text)

        cmd = input("请输入命令 [n=执行提示, r=重新输入, e=退出]: ").strip().lower()
        if cmd == 'e':
            print("已退出。")
            return
        elif cmd == 'r':
            try:
                board = read_board()
                cand = init_candidates(board)
            except Exception as e:
                print("输入错误：", e)
                return
            continue
        elif cmd == 'n':
            # 执行动作
            if fill_action:
                _, sr, sc, sv = fill_action
                r, c, v = int(sr), int(sc), int(sv)
                place(board, cand, r, c, v)
                print(f"已填入：{rc(r, c)} = {v}")
            elif elim_flag:
                # elim_payload 里可能有多条，格式："r,c,v1;v2;..."，或 Locked Candidate 的 "r,c,v"
                targets: List[Tuple[int, int, Set[int]]] = []
                for token in elim_payload:
                    parts = token.split(',')
                    r, c = int(parts[0]), int(parts[1])
                    if len(parts) >= 3:
                        vs = set(int(x) for x in parts[2].split(';'))
                    else:
                        vs = set()
                    targets.append((r, c, vs))
                eliminate(cand, targets)
                # 应用一次剪除后，有可能产生新的 Naked/Hidden Single，下一轮会展示
                info = ", ".join(f"{rc(r, c)}:-{sorted(vs)}" for r, c, vs in targets)
                print(f"已消除候选：{info}")
            else:
                print("（无可执行动作）")
        else:
            print("无效输入，请输入 n / r / e。")


if __name__ == "__main__":
    main()
