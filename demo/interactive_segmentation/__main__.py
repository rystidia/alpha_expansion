import sys

from PyQt6.QtWidgets import QApplication

from .app import ExpansionApp


def main():
    app = QApplication(sys.argv)
    window = ExpansionApp()
    window.show()
    sys.exit(app.exec())


main()
