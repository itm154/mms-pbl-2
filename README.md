# SEIRP Misinformation Model Simulation

Ensure you have [uv](https://docs.astral.sh/uv/) installed

1. Create venv
    ```sh
    uv venv
    ```

2. Activate venv
    - Windows
        ```ps
        .venv\Scripts\activate
        ```
    - Linux/MacOS
        ```sh
        source .venv/bin/activate
        ```
3. Install packages
    ```sh
    uv sync
    ```

4. Run simulation WebUI or generate scenario plots
    - WebUI:
        ```sh
        streamlit run webui.py
        ```
    - Plot all scenario combinations:
        ```sh
        python plot.py
        ```
