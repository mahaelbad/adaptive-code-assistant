"""
Adaptive Code Assistant - Streamlit Application

User-facing interface for:
1. Code Generation
2. Code Explanation

The UI hides internal technical details such as:
- Router classification
- Retrieval relevance
- Generation mode
- Number of retrieved documents
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

from src.assistant import AdaptiveCodeAssistant


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adaptive Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .result-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .helper-text {
        color: #777;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "assistant" not in st.session_state:
    st.session_state.assistant = None

if "result" not in st.session_state:
    st.session_state.result = None

if "run_output" not in st.session_state:
    st.session_state.run_output = None

if "run_error" not in st.session_state:
    st.session_state.run_error = None

if "request_type" not in st.session_state:
    st.session_state.request_type = "Generate Code"

# This value is used to create fresh widget keys
# whenever the user clicks "New Request".
if "form_version" not in st.session_state:
    st.session_state.form_version = 0


# ============================================================
# LOAD ASSISTANT
# ============================================================

@st.cache_resource
def load_assistant():
    """
    Create the Adaptive Code Assistant once
    and reuse it across Streamlit reruns.
    """

    return AdaptiveCodeAssistant()


try:

    if st.session_state.assistant is None:
        st.session_state.assistant = load_assistant()

except Exception as exc:

    st.error(
        "Unable to initialize the Adaptive Code Assistant."
    )

    st.exception(exc)

    st.stop()


assistant = st.session_state.assistant


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_python_code(response: str) -> str:
    """
    Extract Python code from an LLM response.

    Supports:

    ```python
    code
    ```

    or:

    ```
    code
    ```

    If no code block is found, the complete response
    is returned.
    """

    if not response:
        return ""

    # Try Python code block first.
    match = re.search(
        r"```python\s*(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    # Fallback to generic code block.
    match = re.search(
        r"```\s*(.*?)```",
        response,
        flags=re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return response.strip()


def extract_explanation(
    response: str,
) -> str:
    """
    Extract the explanation part from an LLM response.

    If the response contains:

    Explanation:
    ...

    the text after Explanation is returned.

    Otherwise the complete response is returned.
    """

    if not response:
        return ""

    match = re.search(
        r"Explanation\s*:?\s*(.*)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:

        explanation = match.group(1).strip()

        # Remove code blocks if the model repeated them.
        explanation = re.sub(
            r"```python.*?```",
            "",
            explanation,
            flags=re.IGNORECASE | re.DOTALL,
        )

        explanation = re.sub(
            r"```.*?```",
            "",
            explanation,
            flags=re.DOTALL,
        )

        return explanation.strip()

    return response.strip()


def run_python_code(code: str):
    """
    Execute generated Python code locally.

    IMPORTANT:
    This is suitable for local development/testing only.

    For production, generated code should be executed
    inside an isolated sandbox/container.
    """

    if not code or not code.strip():
        return "", "No Python code was provided."

    temp_file = None

    try:

        # Create temporary Python file.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as file:

            file.write(code)

            temp_file = file.name

        # Execute the temporary file.
        process = subprocess.run(
            [
                sys.executable,
                temp_file,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        # Program failed.
        if process.returncode != 0:

            return (
                stdout,
                stderr
                or f"Process exited with code {process.returncode}.",
            )

        # Program succeeded.
        return stdout, None

    except subprocess.TimeoutExpired:

        return (
            "",
            "Execution stopped because the code exceeded the 5-second timeout.",
        )

    except Exception as exc:

        return (
            "",
            str(exc),
        )

    finally:

        # Delete temporary file.
        if temp_file:

            try:

                Path(temp_file).unlink(
                    missing_ok=True
                )

            except Exception:
                pass


def clear_request():
    """
    Clear the current request and generated result.

    IMPORTANT:
    We do NOT directly modify the value of a Streamlit
    widget after it has been instantiated.

    Instead, we increment form_version so Streamlit creates
    completely new widgets on the next rerun.
    """

    st.session_state.result = None
    st.session_state.run_output = None
    st.session_state.run_error = None

    st.session_state.form_version += 1


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Adaptive Code Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Generate Python solutions or understand your existing code.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODE SELECTION
# ============================================================

request_type = st.radio(
    "What would you like to do?",
    [
        "Generate Code",
        "Explain Code",
    ],
    horizontal=True,
    key="request_type",
)


st.divider()


# ============================================================
# GENERATE CODE
# ============================================================

if request_type == "Generate Code":

    st.markdown(
        "### ✨ Generate Code"
    )

    st.markdown(
        """
        Describe what you want to build and the assistant
        will generate a Python solution for you.
        """
    )

    query = st.text_area(
        "Your request",
        placeholder=(
            "Example:\n"
            "Write a Python function called find_max(numbers) "
            "that returns the largest number in a list."
        ),
        height=150,
        key=f"query_input_{st.session_state.form_version}",
    )

    st.markdown(
        '<div class="helper-text">'
        "Tip: Be specific about the input, output, and expected behavior."
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(
        [1, 5]
    )

    with col1:

        generate_clicked = st.button(
            "✨ Generate",
            type="primary",
            use_container_width=True,
        )

    with col2:

        new_request_clicked = st.button(
            "🔄 New Request",
            use_container_width=True,
        )

    if new_request_clicked:

        clear_request()

        st.rerun()

    if generate_clicked:

        if not query.strip():

            st.warning(
                "Please describe what you want to build."
            )

        else:

            with st.spinner(
                "Generating your solution..."
            ):

                try:

                    result = assistant.run(
                        query=query.strip(),
                        code="",
                    )

                    st.session_state.result = result

                    # Clear old execution result.
                    st.session_state.run_output = None
                    st.session_state.run_error = None

                except Exception as exc:

                    st.session_state.result = None

                    st.error(
                        "Sorry, something went wrong while "
                        "processing your request."
                    )

                    st.exception(exc)


# ============================================================
# EXPLAIN CODE
# ============================================================

else:

    st.markdown(
        "### 💡 Explain Code"
    )

    st.markdown(
        """
        Paste your Python code and ask the assistant
        to explain it clearly.
        """
    )

    query = st.text_area(
        "What would you like to know?",
        placeholder=(
            "Example:\n"
            "Explain this code step by step and tell me "
            "its time complexity."
        ),
        height=120,
        key=f"query_input_{st.session_state.form_version}",
    )

    code = st.text_area(
        "Your Python code",
        placeholder=(
            "Paste your Python code here..."
        ),
        height=300,
        key=f"code_input_{st.session_state.form_version}",
    )

    col1, col2 = st.columns(
        [1, 5]
    )

    with col1:

        explain_clicked = st.button(
            "💡 Explain",
            type="primary",
            use_container_width=True,
        )

    with col2:

        new_request_clicked = st.button(
            "🔄 New Request",
            use_container_width=True,
        )

    if new_request_clicked:

        clear_request()

        st.rerun()

    if explain_clicked:

        if not query.strip():

            st.warning(
                "Please enter your question."
            )

        elif not code.strip():

            st.warning(
                "Please paste the code you want to explain."
            )

        else:

            with st.spinner(
                "Analyzing your code..."
            ):

                try:

                    result = assistant.run(
                        query=query.strip(),
                        code=code.strip(),
                    )

                    st.session_state.result = result

                except Exception as exc:

                    st.session_state.result = None

                    st.error(
                        "Sorry, something went wrong while "
                        "processing your request."
                    )

                    st.exception(exc)


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.result


if result:

    st.divider()

    # ========================================================
    # CODE GENERATION RESULT
    # ========================================================

    if result["route"] == "code_generation":

        response = result.get(
            "response",
            "",
        )

        generated_code = extract_python_code(
            response
        )

        explanation = extract_explanation(
            response
        )

        # ----------------------------------------------------
        # SOLUTION
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-title">💻 Solution</div>',
            unsafe_allow_html=True,
        )

        # st.code automatically provides a Copy button.
        st.code(
            generated_code,
            language="python",
        )

        st.markdown(
            '<div class="helper-text">'
            "📋 Use the copy button in the code block to copy the solution."
            "</div>",
            unsafe_allow_html=True,
        )

        st.write("")

        # ----------------------------------------------------
        # RUN CODE
        # ----------------------------------------------------

        run_clicked = st.button(
            "▶️ Run Code",
            type="secondary",
        )

        if run_clicked:

            with st.spinner(
                "Running code..."
            ):

                output, error = run_python_code(
                    generated_code
                )

                st.session_state.run_output = output
                st.session_state.run_error = error

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        if (
            st.session_state.run_output is not None
            or st.session_state.run_error is not None
        ):

            st.markdown(
                '<div class="result-title">🖥️ Output</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.run_error:

                st.error(
                    st.session_state.run_error
                )

            elif st.session_state.run_output:

                st.code(
                    st.session_state.run_output,
                    language="text",
                )

                st.success(
                    "Code executed successfully."
                )

            else:

                st.info(
                    "Code executed successfully, "
                    "but it did not produce any output."
                )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        if explanation:

            st.markdown(
                '<div class="result-title">📝 Explanation</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                explanation
            )

    # ========================================================
    # CODE EXPLANATION RESULT
    # ========================================================

    elif result["route"] == "code_explanation":

        st.markdown(
            '<div class="result-title">📝 Explanation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            result.get(
                "response",
                "",
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:13px;">
        Adaptive Code Assistant • AI-powered Python assistance
    </div>
    """,
    unsafe_allow_html=True,
)