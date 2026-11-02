from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# Overridable so the container can reach the api service by name.
API_URL = os.environ.get("TICKETINTEL_API_URL", "http://localhost:8000")
REPORTS = Path("reports")

EXAMPLES = {
    "Clear - billing": (
        "Double charged for my subscription",
        "My monthly subscription was charged twice this billing cycle. "
        "Please refund the duplicate charge on invoice 48219.",
    ),
    "Ambiguous - product vs technical": (
        "Reports look wrong after the update",
        "Since the latest release some numbers in the dashboard do not match "
        "what we exported last week. Not sure if this is a bug or a setting.",
    ),
}


def triage(subject: str, body: str, summarize: bool) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/triage",
        json={"subject": subject, "body": body, "summarize": summarize},
        timeout=300,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def highlight_unsupported(text: str, unsupported: list[str]) -> str:
    # Mark entities the deterministic check could not ground in the ticket.
    marked = text
    for entity in unsupported:
        marked = marked.replace(entity, f"**:red[{entity}]**")
    return marked


def render_triage_page() -> None:
    st.header("Triage a ticket")
    choice = st.selectbox("Start from an example", ["(blank)", *EXAMPLES])
    default_subject, default_body = EXAMPLES.get(choice, ("", ""))

    subject = st.text_input("Subject", value=default_subject)
    body = st.text_area("Body", value=default_body, height=180)
    want_summary = st.checkbox("Also summarise (needs the local model)", value=False)

    if not st.button("Triage"):
        return
    if not (subject.strip() or body.strip()):
        st.warning("Enter a subject or a body first.")
        return

    result = triage(subject, body, want_summary)
    action = result["action"]
    st.subheader("Decision")
    (st.success if action == "auto_route" else st.warning)(f"{action} - {result['reason']}")
    left, right = st.columns(2)
    left.metric("Queue", result["queue"])
    left.metric("Priority", result["priority"])
    right.metric("Confidence", f"{result['confidence']:.3f}")
    right.write("Candidate queues")
    right.write(result["candidates"])

    if result.get("summary_error"):
        st.info(f"Summary unavailable: {result['summary_error']}")
    elif result.get("summary"):
        st.subheader("Summary")
        unsupported = (result.get("faithfulness") or {}).get("unsupported_entities", [])
        for field, value in result["summary"].items():
            if not value:
                continue
            text = ", ".join(value) if isinstance(value, list) else str(value)
            st.markdown(f"**{field}**: {highlight_unsupported(text, unsupported)}")
        if unsupported:
            st.error(f"Ungrounded entities: {', '.join(unsupported)}")
        else:
            st.caption("Every number, address and code in this summary appears in the ticket.")


def render_evaluation_page() -> None:
    st.header("Evaluation")
    for name in ("calibration_queue.json", "summarization_eval.json"):
        path = REPORTS / "interim" / name
        if not path.exists():
            st.info(f"{name} has not been generated yet.")
            continue
        st.subheader(name)
        st.json(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    st.set_page_config(page_title="Ticket triage console", layout="wide")
    page = st.sidebar.radio("Page", ["Triage", "Evaluation"])
    if page == "Triage":
        render_triage_page()
    else:
        render_evaluation_page()


if __name__ == "__main__":
    main()
