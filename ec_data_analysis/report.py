#!/usr/bin/env python3

from markdown_pdf import MarkdownPdf, Section
from typing import List, Optional


class Report:
    _markdown_texts: List[str]
    _pdf: MarkdownPdf

    def __init__(self, title: str) -> None:
        self._markdown_texts = []
        self._pdf = MarkdownPdf(toc_level=2)
        self._pdf.meta["title"] = title
        self._markdown_texts.append(f"# {title}")

    def addHeading(self, name: str, lvl: int) -> None:
        """
        Adds a Heading of given Level to the Report.

        Args:
            name (str): Name of the heading.
            lvl (int): Positive int indicating the level of the heading.

        Raises:
            ValueError: If level is 0 (only allowed for title) or negative
        """
        if lvl < 0:
            raise ValueError("Level cannot be negative")
        if lvl == 0:
            raise ValueError("Only Title may have Level 0")
        self._markdown_texts.append((lvl + 1) * "#" + " " + name)

    # def addSection(self, sectionName: str) -> None:
    #     if len(self._markdown_texts) > 0:
    #         combined_markdown_text: str = "\n\n".join(self._markdown_texts)
    #         self._pdf.add_section(Section(combined_markdown_text))
    #     self._markdown_texts

    def putFig(self, title: str, filePath: Optional[str]) -> None:
        match filePath:
            case str():
                self._markdown_texts.append(f"![{title}]({filePath})")
            case None:
                raise ValueError(
                    "FilePath passed to Report#putFig(str, Optional[str]) is None."
                )

    def dump(self, text: str) -> None:
        self._markdown_texts.append(text)

    def saveToPdf(self, fileName: str) -> None:
        combined_markdown_text: str = "\n\n".join(self._markdown_texts)
        self._pdf.add_section(Section(combined_markdown_text))
        self._markdown_texts = []
        self._pdf.save(fileName)
