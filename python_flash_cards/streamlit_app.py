import streamlit as st
import pandas as pd
import flashcards_core as fc

st.set_page_config(page_title="D&C Flashcards", layout="centered")

# ---------- Session State ----------
def init_state():
    defaults = {
        "username": None,
        "mode": None,               # "learn" | "review" | "targeted"
        "threshold_raw": "80",
        "num_chapters": 20,
        "start_chapter": 1,
        "flashcards_df": None,
        "user_scores": None,
        "all_scores": None,
        "deck": [],                # current round cards
        "missed": [],              # missed this round
        "idx": 0,
        "session_correct": 0,
        "session_incorrect": 0,
        "finished": False,
        "feedback": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------- Login ----------
st.title("📖 Doctrine & Covenants Flashcards")

if st.session_state.username is None:
    username = st.text_input("Enter your username:", value="")
    if st.button("Start"):
        if username.strip():
            st.session_state.username = username.strip().lower()
            # Expose to core for edge-case inserts
            fc.username_global = st.session_state.username

            # Load cards & scores
            df = fc.load_flashcards()
            st.session_state.flashcards_df = df
            user_scores, all_scores = fc.load_or_create_score_tracker(st.session_state.username, df)
            st.session_state.user_scores = user_scores
            st.session_state.all_scores = all_scores
            st.rerun()
        else:
            st.warning("Please enter a username to continue.")

# ---------- Mode Selection ----------
elif st.session_state.mode is None:
    st.subheader(f"Welcome, **{st.session_state.username}** 👋")

    # Let the user choose how they want to answer (applies to the next session they start)
    st.session_state["answer_freeform"] = st.toggle(
        "Answer with a short summary (local NLP grading)",
        value=st.session_state.get("answer_freeform", False),
        help="If on, you'll see a chapter number and must type a brief summary. We'll grade it with TF-IDF cosine locally."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Learn")
        st.session_state.num_chapters = st.number_input(
            "Number of sections to learn",
            value=st.session_state.num_chapters,
            key="number_input",
        )
        if st.button("Start Learn"):
            st.session_state.mode = "learn"
            st.session_state.deck = fc.build_deck_learn(st.session_state.flashcards_df, st.session_state.num_chapters)
            st.session_state.idx = 0
            st.session_state.missed = []
            st.session_state.session_correct = 0
            st.session_state.session_incorrect = 0
            st.session_state.finished = (len(st.session_state.deck) == 0)
            st.rerun()

    with col2:
        st.write("### Review")
        st.session_state.threshold_raw = st.text_input(
            "Accuracy threshold (e.g., 80 or 0.8)",
            value=st.session_state.threshold_raw,
            key="threshold_input",
        )
        if st.button("Start Review"):
            st.session_state.mode = "review"
            thr = fc.get_accuracy_threshold(st.session_state.threshold_raw)
            st.session_state.deck = fc.build_deck_review(
                st.session_state.flashcards_df, st.session_state.user_scores, thr
            )
            st.session_state.idx = 0
            st.session_state.missed = []
            st.session_state.session_correct = 0
            st.session_state.session_incorrect = 0
            st.session_state.finished = (len(st.session_state.deck) == 0)
            if st.session_state.finished:
                st.info(f"🎉 Nothing to review below {int(thr*100)}% accuracy.")
            else:
                st.rerun()

    with col3:
        st.write("### Targeted (10 in a row)")
        st.session_state.start_chapter = st.number_input(
            "Start chapter", min_value=1, value=int(st.session_state.start_chapter), step=1
        )
        if st.button("Start Targeted"):
            st.session_state.mode = "targeted"
            st.session_state.deck = fc.build_deck_targeted(
                st.session_state.flashcards_df, int(st.session_state.start_chapter)
            )
            st.session_state.idx = 0
            st.session_state.missed = []
            st.session_state.session_correct = 0
            st.session_state.session_incorrect = 0
            st.session_state.finished = (len(st.session_state.deck) == 0)
            if st.session_state.finished:
                st.warning("No chapters found for that starting chapter.")
            else:
                st.rerun()

# ---------- Gameplay ----------
else:
    if st.session_state.finished or not st.session_state.deck:
        st.success("🎉 Session complete!")

        # Save stats
        fc.save_score_tracker(st.session_state.user_scores, st.session_state.all_scores, st.session_state.username)
        fc.update_daily_stats(
            st.session_state.username,
            st.session_state.session_correct,
            st.session_state.session_incorrect,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Correct", st.session_state.session_correct)
        with c2:
            st.metric("Incorrect", st.session_state.session_incorrect)

        st.divider()
        st.caption("Your chapter stats:")
        st.dataframe(
            fc.calculate_accuracy(st.session_state.user_scores).sort_values("Total", ascending=False),
            use_container_width=True,
        )

        if st.button("Play Again"):
            keep_user = st.session_state.username
            st.session_state.clear()
            init_state()
            st.session_state.username = keep_user
            fc.username_global = keep_user
            # Reload fresh
            df = fc.load_flashcards()
            st.session_state.flashcards_df = df
            user_scores, all_scores = fc.load_or_create_score_tracker(keep_user, df)
            st.session_state.user_scores = user_scores
            st.session_state.all_scores = all_scores
            st.rerun()

    else:
        deck = st.session_state.deck
        i = st.session_state.idx
        card = deck[i]

        st.subheader(f"Mode: {st.session_state.mode.title()}")

        # Show either the Summary (numeric-answer mode) or the Chapter (free-form summary mode)
        if st.session_state.get("answer_freeform"):
            st.write(f"Chapter:** {card['Chapter']}")
            prompt_label = "Type a short summary in your own words:"
        else:
    # List of fields to display if they have data
            fields = [
                "Summary",
                "Date",
                "Recipient",
                "Verse",
                "Context",
                "Content",
                "Controversy",
                "Consequences",
                "Theme",
                "Location",
                "Approximate"
            ]

            for field in fields:
                value = card.get(field)
                if pd.notna(value) and value not in (None, "", "null","nan"):
                    st.write(f"**{field}:** {value}")
            prompt_label = "Enter chapter:"

        col_ok, col_skip = st.columns([3, 1])

        # --- Submit (inside a form so Enter works) ---
        with col_ok:
            with st.form(key=f"form_{i}"):
                answer = st.text_input(prompt_label, key=f"answer_{i}")
                submitted = st.form_submit_button("Submit")

                if submitted:
                    # Decide how to grade based on the toggle
                    if st.session_state.get("answer_freeform"):
                        # Free-form: grade locally via TF-IDF
                        verdict = fc.evaluate_freeform_local(
                            reference_summary=card["Summary"],
                            user_summary=answer,
                            tfidf_threshold=0.48 # Tune as needed
                        )
                        is_correct = verdict["correct"]
                        st.session_state.feedback = (
                            f"{'✅ Correct' if is_correct else '❌ Incorrect'} "
                            f"• method={verdict['method']} • score={verdict['score']}"
                        )
                    else:
                        # Numeric Check
                        is_correct = fc.evaluate_answer(card["Chapter"], answer)
                        st.session_state.feedback = (
                            "✅ Correct!" if is_correct else f"❌ Incorrect. Correct: {card['Chapter']}"
                        )

                    # Update stats
                    st.session_state.user_scores = fc.update_score(
                        st.session_state.user_scores, card["Chapter"], is_correct
                    )

                    if is_correct:
                        st.session_state.session_correct += 1
                    else:
                        st.session_state.session_incorrect += 1
                        st.session_state.missed.append(card)

                    # Advance
                    st.session_state.idx += 1

                    # End-of-round handling
                    if st.session_state.idx >= len(st.session_state.deck):
                        if st.session_state.missed:
                            fc.shuffle_records(st.session_state.missed)
                            st.session_state.deck = st.session_state.missed
                            st.session_state.missed = []
                            st.session_state.idx = 0
                        else:
                            st.session_state.finished = True

                    st.rerun()

        # --- Skip button (outside form) ---
        with col_skip:
            if st.button("Skip"):
                st.session_state.user_scores = fc.update_score(
                    st.session_state.user_scores, card["Chapter"], False
                )
                st.session_state.session_incorrect += 1
                st.session_state.missed.append(card)
                st.session_state.idx += 1

                if st.session_state.idx >= len(st.session_state.deck):
                    if st.session_state.missed:
                        fc.shuffle_records(st.session_state.missed)
                        st.session_state.deck = st.session_state.missed
                        st.session_state.missed = []
                        st.session_state.idx = 0
                    else:
                        st.session_state.finished = True

                st.rerun()

        # --- Feedback display ---
        if st.session_state.feedback:
            st.info(st.session_state.feedback)
