#!/usr/bin/env python3

from markdown_pdf import MarkdownPdf, Section
from typing import List, Optional, TextIO


class Report:
    _text_queue: List[str]
    _pdf: MarkdownPdf

    def __init__(self, title: str) -> None:
        self._text_queue = []
        self._pdf = MarkdownPdf(toc_level=2)
        self._pdf.meta["title"] = title
        self._text_queue.append(f"# {title}")
        self._currentSectionName = None

    def addSection(self, sectionName: str) -> None:
        """
        Logically adds a section by concluding the last one and adding a lvl 1 heading with the section name.
        """
        self._concludeLastSection()
        self.addHeading(sectionName, 1)

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
        self._text_queue.append((lvl + 1) * "#" + " " + name)

    def dump(self, text: str) -> None:
        """
        Dumps the given text into the text queue, i.e. within the currently open section.
        """
        self._text_queue.append(text)

    def dumpFile(self, filePath: str) -> None:
        """
        Dumps the text from the given file into the text queue, i.e. within the currently open section.
        """
        try:
            file: TextIO = open(filePath, "r", encoding="utf-8")
            text: str = file.read()
            self._text_queue.append(text)
        except FileNotFoundError:
            print(f"Error: Filepath '{filePath}' not found.")

    def putFigs(self, filePath1: Optional[str], filePath2: Optional[str]) -> None:
        """
        Puts the figures on the given paths next to each other into the document within the currently open section.
        """
        match filePath1:
            case str():
                match filePath2:
                    case str():
                        self._text_queue.append(
                            f'<p float="left"> <img src="{filePath1}" width="220" /> <img src="{filePath2}" width="220" /> </p>'
                        )
                        # self._text_queue.append(f"![leftFig]({filePath1}) ![rightFig]({filePath2})")

    def putFig(self, title: str, filePath: Optional[str]) -> None:
        """
        Puts the figure on the given path into the document within the currently open section.
        """
        match filePath:
            case str():
                self._text_queue.append(f"![{title}]({filePath})")
            case None:
                raise ValueError(
                    "FilePath passed to Report#putFig(str, Optional[str]) is None."
                )

    def saveToPdf(self, fileName: str) -> None:
        self._concludeLastSection()
        self._pdf.save(fileName)

    def _concludeLastSection(self) -> None:
        """
        Adds all queued texts to a new section and clears the queue.
        """
        if len(self._text_queue) > 0:
            combined_markdown_text: str = "\n\n".join(self._text_queue)
            self._pdf.add_section(Section(combined_markdown_text))
            self._text_queue = []
