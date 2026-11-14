"""Consistência do rascunho do paper (não compila o LaTeX): macros, citações e arquivos."""

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper" / "khipu"
TEX = (PAPER / "main.tex").read_text()
BIB = (PAPER / "refs.bib").read_text()
REQUIRED_ARXIV_IDS = {
    "1308.3432", "1711.00937", "1701.06538", "2101.03961", "2202.01169", "2310.00811",
    "2409.12136", "2504.12463", "2305.08842", "2410.06424", "2411.19402", "2605.14359",
}  # fmt: skip


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def test_every_result_macro_used_is_defined_in_numbers_tex():
    numbers = "".join(
        (PAPER / rel).read_text()
        for rel in ("tables/numbers.tex", "tables/numbers_val.tex", "tables_tau_teste/numbers.tex")
    )
    defined = set(re.findall(r"\\newcommand\{\\(res[A-Za-z]+)\}", numbers))
    used = set(re.findall(r"\\(res[A-Z][A-Za-z]*)", strip_comments(TEX)))
    assert used, "o texto não usa nenhum macro de resultado"
    assert used <= defined, f"macros usados e não definidos: {sorted(used - defined)}"


def test_cite_keys_exist_in_bib_and_bib_has_only_the_requested_ids():
    keys = set(re.findall(r"^@\w+\{([^,]+),", BIB, re.M))
    cited = {
        k.strip()
        for c in re.findall(r"\\cite\{([^}]*)\}", strip_comments(TEX))
        for k in c.split(",")
    }
    assert cited <= keys, f"citações sem entrada no .bib: {sorted(cited - keys)}"
    assert set(re.findall(r"eprint\s*=\s*\{([\d.]+)\}", BIB)) == REQUIRED_ARXIV_IDS
    assert len(keys) == len(REQUIRED_ARXIV_IDS)


def test_included_files_exist():
    body = strip_comments(TEX)
    for rel in re.findall(r"\\input\{([^}]+)\}", body) + re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body
    ):
        assert (PAPER / rel).exists(), f"arquivo ausente: {rel}"


def test_every_literature_claim_block_is_flagged():
    # o pedido exige \todo{} em toda frase de literatura a conferir: o texto tem de ter vários
    assert len(re.findall(r"\\todo\{", strip_comments(TEX))) >= 4
