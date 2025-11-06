import pandas as pd
import random
import os
from datetime import date
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# ---------- Files ----------
FLASHCARD_FILE = "d_and_c_chapters.csv"
SCORE_FILE = "chapter_stats.csv"
DAILY_STATS_FILE = "daily_stats.csv"

# Used by update_score to know which user to attach when it needs to insert a missing chapter row.
username_global = None

def get_username():
    """
    Prompt user for a username.
    Returns:
        str: Cleaned lowercase username.
    """
    username = input("Enter your username: ").strip().lower()
    return username or "default"   

def load_flashcards():
    """
    Load and shuffle flashcards from the CSV file.

    Returns:
        pd.DataFrame: A shuffled DataFrame with columns 'Chapter' and 'Summary'.
    """
    df = pd.read_csv(FLASHCARD_FILE)
    df['Chapter'] = df['Chapter'].astype(str)
    return df

def load_or_create_score_tracker(username: str, flashcards: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load chapter score file (all users) and return:
    - user_scores: a copy filtered to this username (with all chapters present)
    - all_scores: the full table (for merging & saving later)
    """
    if os.path.exists(SCORE_FILE):
        all_scores = pd.read_csv(SCORE_FILE)
        for col in ("username", "Chapter"):
            if col in all_scores.columns:
                all_scores[col] = all_scores[col].astype(str).str.strip()
    else:
        all_scores = pd.DataFrame(columns=["username", "Chapter", "Correct", "Incorrect", "Total"])

    user_scores = all_scores[all_scores["username"] == username].copy()

    # Ensure every chapter exists for the user
    user_scores = ensure_user_rows_for_all_chapters(user_scores, flashcards, username)

    return user_scores, all_scores


def ensure_user_rows_for_all_chapters(user_scores: pd.DataFrame, flashcards: pd.DataFrame, username: str) -> pd.DataFrame:
    """
    If user's score tracker is missing any chapters, add zeroed rows.
    """
    have = set(user_scores["Chapter"].astype(str))
    want = set(flashcards["Chapter"].astype(str))
    missing = list(want - have)
    if missing:
        to_add = pd.DataFrame({
            "username": [username] * len(missing),
            "Chapter": missing,
            "Correct": 0,
            "Incorrect": 0,
            "Total": 0
        })
        user_scores = pd.concat([user_scores, to_add], ignore_index=True)
    return user_scores


def save_score_tracker(user_scores: pd.DataFrame, all_scores: pd.DataFrame, username: str) -> None:
    """
    Merge updated user score data back into the full file and save it.
    """
    all_scores = all_scores[all_scores["username"] != username]
    all_scores = pd.concat([all_scores, user_scores], ignore_index=True)
    all_scores.to_csv(SCORE_FILE, index=False)


def update_daily_stats(username: str, correct: int, incorrect: int) -> None:
    """
    Track per-user, per-day performance in DAILY_STATS_FILE.
    """
    today = date.today().isoformat()
    total = correct + incorrect

    if os.path.exists(DAILY_STATS_FILE):
        daily_df = pd.read_csv(DAILY_STATS_FILE)
        if "username" in daily_df.columns:
            daily_df["username"] = daily_df["username"].astype(str)
    else:
        daily_df = pd.DataFrame(columns=["username", "date", "correct", "incorrect", "total"])

    match = (daily_df["username"] == username) & (daily_df["date"] == today)
    if match.any():
        idx = daily_df[match].index[0]
        daily_df.at[idx, "correct"] += correct
        daily_df.at[idx, "incorrect"] += incorrect
        daily_df.at[idx, "total"] += total
    else:
        new_row = pd.DataFrame([{
            "username": username,
            "date": today,
            "correct": correct,
            "incorrect": incorrect,
            "total": total
        }])
        daily_df = pd.concat([daily_df, new_row], ignore_index=True)

    daily_df.to_csv(DAILY_STATS_FILE, index=False)

# ---------- Stats / Accuracy ----------
def calculate_accuracy(user_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add or update an 'accuracy' column in user_scores.
    Accuracy = Correct / (Correct + Incorrect).
    """
    user_scores = user_scores.copy()
    user_scores["accuracy"] = user_scores.apply(
        lambda r: (r["Correct"] / (r["Correct"] + r["Incorrect"])) if (r["Correct"] + r["Incorrect"]) > 0 else 0.0,
        axis=1
    )
    return user_scores


def get_accuracy_threshold(raw_input: str) -> float:
    """
    Normalize threshold input. Accepts '0.8' or '80' -> returns 0.8.
    """
    try:
        val = float(raw_input)
        return val / 100.0 if val > 1 else val
    except ValueError:
        return 0.8

# ---------- Deck Builders (no UI) ----------
def build_deck_learn(flashcards: pd.DataFrame) -> list[dict]:
    """
    Learn mode: all chapters, shuffled.
    """
    deck = flashcards.to_dict("records")
    shuffle_records(deck)
    return deck


def build_deck_review(flashcards: pd.DataFrame, user_scores: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Review mode: chapters whose accuracy is below threshold.
    """
    scored = calculate_accuracy(user_scores)
    low = set(scored[scored["accuracy"] < threshold]["Chapter"].astype(str))
    deck_df = flashcards[flashcards["Chapter"].astype(str).isin(low)].copy()
    deck = deck_df.to_dict("records")
    shuffle_records(deck)
    return deck


def build_deck_targeted(flashcards: pd.DataFrame, start_chapter: int) -> list[dict]:
    """
    Targeted mode: block of up to 10 consecutive chapters starting at start_chapter.
    """
    cards = flashcards.copy()
    cards["Chapter_num"] = cards["Chapter"].astype(int)
    max_chapter = cards["Chapter_num"].max()
    end_chapter = min(start_chapter + 9, max_chapter)
    block = cards[
        (cards["Chapter_num"] >= start_chapter) & 
        (cards["Chapter_num"] < end_chapter)
    ]
    deck = block.sort_values("Chapter_num").to_dict("records")
    shuffle_records(deck)
    return deck


def shuffle_records(records: list[dict]) -> None:
    """
    In-place shuffle of a list of dicts.
    """
    random.shuffle(records)


# ---------- Answer Checking / Score ----------
def evaluate_answer(chapter: str, user_input: str) -> bool:
    """
    Compare user input to the expected chapter number (as strings).
    """
    return str(user_input).strip() == str(chapter).strip()


def update_score(user_scores: pd.DataFrame, chapter: str, correct: bool) -> pd.DataFrame:
    """
    Update in-memory score tracker for a chapter. Inserts the row if missing.
    """
    chapter = str(chapter)
    match = (user_scores["Chapter"].astype(str) == chapter)

    if not match.any():
        # Insert if missing (edge case safety)
        new_row = {
            "username": username_global,
            "Chapter": chapter,
            "Correct": int(bool(correct)),
            "Incorrect": int(not correct),
            "Total": 1
        }
        user_scores.loc[len(user_scores)] = new_row
        return user_scores

    idx = user_scores[match].index[0]
    user_scores.at[idx, "Total"] += 1
    if correct:
        user_scores.at[idx, "Correct"] += 1
    else:
        user_scores.at[idx, "Incorrect"] += 1

    return user_scores


##############################
# Implement Logic to allow for sentence parsing and matching
##############################

def _normalize_text(s: str) -> str:
    """
    Light normalization: lowercase, collapse whitespace, strip punctuation.
    """
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def similarity_tfidf(reference_summary: str, user_summary: str) -> float:
    """
    Compute TF-IDF cosine similarity between reference and user text.
    Returns float in [0,1].
    """
    ref = _normalize_text(reference_summary)
    usr = _normalize_text(user_summary)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams help short texts
        stop_words="english"    # reduce noise
    )
    X = vectorizer.fit_transform([ref, usr])
    sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
    return float(sim)

def evaluate_freeform_local(reference_summary: str,
                            user_summary: str,
                            tfidf_threshold: float = 0.45) -> dict:
    """
    Grade a free-form user summary locally via TF-IDF cosine.
    Returns:
      { "correct": bool, "method": "tfidf", "score": float }
    """
    score = similarity_tfidf(reference_summary, user_summary)
    return {
        "correct": bool(score >= tfidf_threshold),
        "method": "tfidf",
        "score": round(score, 3),
    }

# if __name__ == "__main__":
#     username = get_username()
#     username_global = username
#     flashcards = load_flashcards()
#     user_scores, all_scores = load_or_create_score_tracker(username, flashcards)

#     mode = input("Select mode (learn/review/targeted): ").strip().lower()

#     if mode == "review" or mode == "r":
#         threshold_input = input("Enter accuracy threshold: ")
#         threshold = get_accuracy_threshold(threshold_input)
#         correct, incorrect = review_mode(flashcards, user_scores, threshold)
#     elif mode == "targeted" or mode == "t":
#         correct, incorrect = targeted_review_mode(flashcards, user_scores)
#     else:
#         print("\n🧠 Starting learn mode with all chapters...")
#         correct, incorrect = start_flashcards(flashcards, user_scores)
    
#     show_summary(user_scores)
#     save_score_tracker(user_scores, all_scores, username)
#     update_daily_stats(username,correct, incorrect)