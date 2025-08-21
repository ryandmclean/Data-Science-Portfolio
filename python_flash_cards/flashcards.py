import pandas as pd
import random
import os
from datetime import date

FLASHCARD_FILE = "d_and_c_chapters.csv"
SCORE_FILE = "chapter_stats.csv"
DAILY_STATS_FILE = "daily_stats.csv"
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

def load_or_create_score_tracker(username, flashcards):
    """
    Load the chapter score file and return the current user's rows, creating any new rows as needed.

    Args:
        username (str): Current user.
        flashcards (pd.DataFrame): Flashcard data.

    Returns:
        pd.DataFrame: Chapter score tracker for this user.
    """
    if os.path.exists(SCORE_FILE):
        all_scores = pd.read_csv(SCORE_FILE)
        all_scores['Chapter'] = all_scores['Chapter'].astype(str)
        all_scores['username'] = all_scores['username'].astype(str)
    else:
        all_scores = pd.DataFrame(columns=['username', 'Chapter', 'Correct', 'Incorrect', 'Total'])

    user_scores = all_scores[all_scores['username'] == username].copy()

    existing_chapters = set(user_scores['Chapter'])
    new_chapters = set(flashcards['Chapter']) - existing_chapters

    if new_chapters:
        new_rows = pd.DataFrame({
            'username': username,
            'Chapter': list(new_chapters),
            'Correct': 0,
            'Incorrect': 0,
            'Total': 0
        })
        score_df = pd.concat([user_scores, new_rows], ignore_index=True)
    return user_scores, all_scores

def calculate_accuracy(user_scores):
    """
    Add or update an 'accuracy' column in user_scores.
    Accuracy = Correct / (Correct + Incorrect).
    """
    user_scores['accuracy'] = user_scores.apply(
        lambda row: row['Correct'] / (row['Total'])
        if (row['Total']) > 0 else 0.0,
        axis=1
    )
    return user_scores

def update_score(user_scores, chapter, correct):
    """
    Update in-memory score tracker. Add chapter if it doesn't exist yet for the user.
    """
    chapter = str(chapter)

    match = user_scores['Chapter'] == chapter
    if not match.any():
        # Dynamically add chapter if somehow it was missing
        new_row = {
            'username': username_global,  # from global context
            'Chapter': chapter,
            'Correct': int(correct),
            'Incorrect': int(not correct),
            'Total': 1
        }
        user_scores.loc[len(user_scores)] = new_row
        return

    idx = user_scores[match].index[0]
    user_scores.at[idx, 'Total'] += 1
    if correct:
        user_scores.at[idx, 'Correct'] += 1
    else:
        user_scores.at[idx, 'Incorrect'] += 1

def start_flashcards(flashcards, user_scores):
    """
    Run the flashcard quiz loop with reshuffling and review for incorrect answers.

    Args:
        flashcards (pd.DataFrame): The flashcard data.
        user_scores (pd.DataFrame): The score tracking data.
    """
    to_review = flashcards.copy()
    session_correct = 0
    session_incorrect = 0

    while not to_review.empty:
        to_review = to_review.sample(frac=1).reset_index(drop = True) # Shuffle each round
        missed = []

        for _, row in to_review.iterrows():
            print("\nSummary:")
            print(row['Summary'])
            user_input = input("What chapter is this? ")

            is_correct = user_input.strip() == row['Chapter']
            update_score(user_scores, row['Chapter'], is_correct)

            if is_correct:
                print("✅ Correct!")
                session_correct += 1
            else:
                print(f"❌ Incorrect. The correct chapter is {row['Chapter']}")
                session_incorrect += 1
                missed.append(row)
        
        to_review = pd.DataFrame(missed)
    
    print("\n🎉 All cards answered correctly!")
    return session_correct, session_incorrect

def show_summary(user_scores):
    """
    Display the summary statistics for all chapters.

    Args:
        user_scores (pd.DataFrame): The score tracking DataFrame.
    """
    user_scores = calculate_accuracy(user_scores)
    print("\n📊 Chapter Stats:")
    print(user_scores.sort_values(by='Total', ascending=False))

def save_score_tracker(user_scores, all_scores, username):
    """
    Merge updated user score data back into the full file and save it.
    """
    all_scores = all_scores[all_scores['username'] != username]
    all_scores = pd.concat([all_scores, user_scores], ignore_index=True)
    all_scores.to_csv(SCORE_FILE, index=False)
    print(f"\n💾 Stats saved to {SCORE_FILE}")

def update_daily_stats(username, correct, incorrect):
    """
    Track per-user, per-day performance.

    Args:
        username (str): The current user playing
        correct (int): Number correct in this session
        incorrect (int): Number incorrect in this session
    """
    today = date.today().isoformat()
    total = correct + incorrect

    if os.path.exists(DAILY_STATS_FILE):
        daily_df = pd.read_csv(DAILY_STATS_FILE)
        daily_df['username'] = daily_df['username'].astype(str)
    else:
        daily_df = pd.DataFrame(columns=['username', 'date', 'correct', 'incorrect', 'total'])

    match = (daily_df['username'] == username) & (daily_df['date'] == today)

    if match.any():
        idx = daily_df[match].index[0]
        daily_df.at[idx, 'correct'] += correct
        daily_df.at[idx, 'incorrect'] += incorrect
        daily_df.at[idx, 'total'] += total
    else:
        new_row = pd.DataFrame([{
            'username': username,
            'date': today,
            'correct': correct,
            'incorrect': incorrect,
            'total': total
        }])
        daily_df = pd.concat([daily_df, new_row], ignore_index=True)

    daily_df.to_csv(DAILY_STATS_FILE, index=False)
    print(f"📅 Updated daily stats in {DAILY_STATS_FILE}")

def get_accuracy_threshold(raw_input):
    """
    Normalize threshold input from user. Accepts decimal (0.8) or whole number (80).
    
    Args:
        raw_input (str): User-entered accuracy value.

    Returns:
        float: Threshold as a decimal (e.g., 0.8 for 80%)
    """
    try:
        value = float(raw_input)
        if value > 1:
            return value / 100.0
        return value
    except ValueError:
        print("⚠️ Invalid threshold. Defaulting to 0.8 (80%).")
        return 0.8
    
def review_mode(flashcards, user_scores, threshold):
    """
    Filter flashcards based on accuracy threshold and run quiz only on weak areas.

    Args:
        flashcards (pd.DataFrame): The original flashcard set.
        score_df (pd.DataFrame): The score tracker.
        threshold (float): Max allowed accuracy to include in review.
    """
    # Calculate accuracy per chapter
    user_scores['accuracy'] = calculate_accuracy(user_scores)
    low_perf_chapters = user_scores[user_scores['accuracy'] < threshold]['Chapter'].tolist()

    if not low_perf_chapters:
        print(f"\n🎉 All chapters meet or exceed {int(threshold * 100)}% accuracy. Nothing to review!")
        return 0, 0

    review_set = flashcards[flashcards['Chapter'].isin(low_perf_chapters)].reset_index(drop=True)
    print(f"\n📘 Reviewing {len(review_set)} chapter(s) with accuracy < {int(threshold * 100)}%")
    return start_flashcards(review_set, user_scores)



if __name__ == "__main__":
    username = get_username()
    username_global = username
    flashcards = load_flashcards()
    user_scores, all_scores = load_or_create_score_tracker(username, flashcards)

    mode = input("Select mode (learn/review): ").strip().lower()

    if mode == "review":
        threshold_input = input("Enter accuracy threshold: ")
        threshold = get_accuracy_threshold(threshold_input)
        correct, incorrect = review_mode(flashcards, user_scores, threshold)
    else:
        print("\n🧠 Starting learn mode with all chapters...")
        correct, incorrect = start_flashcards(flashcards, user_scores)
    
    show_summary(user_scores)
    save_score_tracker(user_scores, all_scores, username)
    update_daily_stats(username,correct, incorrect)