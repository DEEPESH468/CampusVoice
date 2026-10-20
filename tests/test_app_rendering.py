import pandas as pd
from streamlit.testing.v1 import AppTest


def test_unlabelled_feedback_warning_does_not_block_overview_analytics():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.session_state["feedback"] = pd.DataFrame({
        "comment": ["Teachers are helpful.", "The lab equipment is broken."],
        "department": ["Computer Science", "Engineering"],
        "category_hint": ["teaching", "laboratories"],
    })
    app.session_state["source_name"] = "uploaded rich CSV"
    app.run()

    assert not app.exception
    assert any(metric.label == "Comments analyzed" and metric.value == "2" for metric in app.metric)
    assert len(app.get("plotly_chart")) >= 3
    assert any("Unlabelled feedback" in block.value for block in app.markdown)


def test_sidebar_pages_render_their_own_content_with_feedback_loaded():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.button[0].click().run()  # Load synthetic dashboard feedback.
    expected = {
        "Sentiment Analytics": "Sentiment analytics",
        "Aspect Analysis": "Aspect based opinion analysis",
        "Topic Discovery": "Topic discovery",
        "Department Comparison": "Department comparison",
        "Feedback Explorer": "Feedback explorer",
        "Actionable Insights": "Areas for human review",
        "Train Sentiment Model & Evaluation": "Train Sentiment Model",
        "Reports & Export": "Reports and export",
        "Methodology": "Methodology and responsible use",
    }
    for page, heading in expected.items():
        app.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {[error.message for error in app.exception]}"
        assert any(block.value == heading for block in app.subheader), page
