"""Local review page for "Wrong card?" reports. Run: python -m scripts.review, then open http://127.0.0.1:8932"""
import os
from html import escape
from urllib.parse import quote

import uvicorn
from botocore.exceptions import BotoCoreError
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from pokegrade import review
from pokegrade.pricing import lookup_local_card
from pokegrade.search import search_cards

# Set the photo bucket here instead of in .env, so running the main app locally doesn't save scans
os.environ.setdefault("SCANS_BUCKET", "pokegrade-scans-017421876144-us-east-2-an")

app = FastAPI()


# Show a clear message instead of a server error when the AWS login is missing or expired
@app.exception_handler(BotoCoreError)
def aws_error(request: Request, exc: BotoCoreError):
    return HTMLResponse(
        f'<p style="color:red">AWS error: {escape(str(exc))}</p>'
        "<p>Run <code>aws login</code> in the terminal, then reload this page.</p>",
        status_code=503,
    )


# Turn a card ID into a readable name, set, and number for the page
def describe(card_id):
    if card_id is None:
        return "None of these"
    info = lookup_local_card(card_id)
    if info is None:
        return escape(card_id)
    name, set_name, number = info
    return escape(f"{name} · {set_name} #{number} ({card_id})")


# Show a photo from S3, or a placeholder if it was never uploaded
def photo(key):
    if key is None:
        return "<p>(no photo)</p>"
    return f'<img src="{escape(review.photo_url(key))}" height="300">'


# Build the section for one report, with its photos, both guesses, and the review buttons
def report_html(row):
    report_id, created_at, query, user_card, app_card, gap, photo_key, crop_key = row
    unsure = " ⚠️ unsure" if gap is not None and gap < 0.03 else ""
    gap_text = f"{gap:.3f}" if gap is not None else "?"
    return f"""
    <hr>
    <h3>#{report_id} · {created_at:%Y-%m-%d %H:%M} · typed "{escape(query)}"</h3>
    {photo(photo_key)} {photo(crop_key)}
    <p>App guessed: {describe(app_card)} (gap {gap_text}{unsure})<br>
       User picked: {describe(user_card)}</p>
    <form method="post" action="/approve/{report_id}" style="display:inline">
      <button>Approve</button>
    </form>
    <form method="post" action="/approve/{report_id}" style="display:inline">
      Approve as <input name="card_id" placeholder="svp/176" size="12" required> <button>Go</button>
    </form>
    <form method="post" action="/reject/{report_id}" style="display:inline">
      <button>Reject</button>
    </form>
    """


# Show the review page with every report that still needs a decision
@app.get("/", response_class=HTMLResponse)
def page(q: str = "", error: str = ""):
    pending = review.list_pending()
    found = "".join(f"<li>{describe(c['card_id'])}</li>" for c in search_cards(q)) if q else ""
    return f"""<!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>PokeGrade Review</title></head><body>
    <h1>Reports waiting for review ({len(pending)})</h1>
    {f'<p style="color:red">{escape(error)}</p>' if error else ""}
    <form method="get">Find a card ID: <input name="q" value="{escape(q)}"> <button>Search</button></form>
    <ul>{found}</ul>
    {"".join(report_html(row) for row in pending) or "<p>Nothing to review.</p>"}
    </body></html>"""


# Approve a report and save it as a label, then go back to the review page
@app.post("/approve/{report_id}")
def approve(report_id: int, card_id: str = Form(None)):
    error = review.approve(report_id, card_id.strip() if card_id else None)
    return RedirectResponse(f"/?error={quote(error)}" if error else "/", status_code=303)


# Reject a report, then go back to the review page
@app.post("/reject/{report_id}")
def reject(report_id: int):
    review.reject(report_id)
    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8932)
