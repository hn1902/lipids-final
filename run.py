"""Entry point for the Lipidomics Data Analysis Shiny Application."""
from shiny import run_app

if __name__ == "__main__":
    run_app("app.app:app", port=8000, host="127.0.0.1", reload=False)
