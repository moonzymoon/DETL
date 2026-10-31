"""生成合并全文版 LaTeX (正文 \input 展开 + refs.bib 附后).

用途: 每次改稿后运行一次, 产出 paper/Elsevier_JLPPI_DETL_合并全文版.tex,
整个文件直接复制给外部 AI (DeepSeek 等) 审查——无需拼多个文件.
注意: 该文件是派生物, 不要手改; 改论文请改 sections/ 后重跑本脚本.

运行: cd src && python experiments/make_merged_tex.py
"""
import os
import re

PAPER = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
MAIN = os.path.join(PAPER, "Elsevier_JLPPI_DETL.tex")
OUT = os.path.join(PAPER, "Elsevier_JLPPI_DETL_合并全文版.tex")


def main():
    tex = open(MAIN, encoding="utf-8").read()

    def repl(m):
        path = os.path.join(PAPER, m.group(1) + ".tex")
        body = open(path, encoding="utf-8").read()
        return ("%% ===================== %s =====================\n"
                % m.group(1).replace("/", ".")) + body

    merged = re.sub(r"\\input\{([^}]+)\}", repl, tex)
    bib = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()
    merged = merged.replace(
        "\\bibliographystyle{elsarticle-harv}\n\\bibliography{refs}",
        "%% ============ 参考文献 (refs.bib, author-year 制, 字母序) ============\n"
        "%% 注意: 本刊用作者-年份引用 (Gyasi and Wang, 2022), 文献表按字母序不编号,\n"
        "%% 这是 JLPPI 官方指南规定, 不是缺编号.\n"
        + bib)
    open(OUT, "w", encoding="utf-8").write(merged)
    print(f"写出 {OUT} ({len(merged.splitlines())} 行)")


if __name__ == "__main__":
    main()
