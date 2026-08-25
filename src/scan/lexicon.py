"""Word lists behind the transcript metrics.

Separated from the measurement code so that tuning what counts as a filler, or
adding the administrative vocabulary of a course that talks about Gradescope
instead of Canvas, is a data edit. Every list here is deliberately small and
literal: these feed metrics that are read as rough indicators next to a cohort
percentile, not as classifications anyone should defend individually.
"""

# Spoken fillers. Multi-word entries are matched as phrases before single
# words, so "you know" is one filler and not two.
#
# Deliberately excludes so, right, like, well, okay, actually and anyway. All
# of them are genuine filler tics in some mouths, and all of them are ordinary
# connectives in a technical lecture -- "so we get", "the right subtree",
# "well-ordered". Telling the two apart needs a parser, and counting them
# regardless would make the metric a measure of how much maths is being
# spoken. What is left is the set that is filler nearly every time it occurs.
FILLERS = [
    "you know", "i mean", "sort of", "kind of", "or something",
    "and stuff", "and so forth",
    "um", "uh", "erm", "uhm", "hmm", "mhm",
    "basically", "essentially", "literally", "obviously",
]

# Discourse markers that tell a listener where they are in the lecture.
# These are what a prepared lecture has and a rambled one does not.
SIGNPOSTS = [
    "first", "firstly", "second", "secondly", "third", "thirdly",
    "next", "then", "finally", "lastly", "to begin", "let us start",
    "let's start", "start with", "move on", "moving on", "turn to",
    "recall", "remember that", "as we saw", "last time", "previously",
    "earlier we", "so far", "up to now",
    "the key idea", "the main idea", "the point is", "the important thing",
    "notice that", "note that", "observe that", "the intuition",
    "in summary", "to summarize", "to summarise", "to recap", "recap",
    "in other words", "put another way", "that is to say",
    "for example", "for instance", "such as", "consider", "suppose",
    "the plan", "we will cover", "we'll cover", "today we", "in this lecture",
    "which means", "therefore", "hence", "as a result", "it follows",
]

# Course administration. Distinguishes a lecture that teaches from one that is
# largely logistics -- the single strongest signal of whether a recording is
# useful to somebody outside the section.
ADMIN_TERMS = [
    "homework", "assignment", "problem set", "pset", "due date", "due on",
    "deadline", "extension", "late day", "late days", "submit", "submission",
    "gradescope", "canvas", "piazza", "blackboard", "autolab", "moodle",
    "exam", "midterm", "final exam", "quiz", "grading", "grade", "graded",
    "rubric", "curve", "office hours", "recitation", "ta ", "tas ",
    "syllabus", "attendance", "enrol", "enroll", "waitlist", "drop deadline",
    "extra credit", "makeup", "regrade", "academic integrity", "collaboration policy",
]

OPENING_CUES = [
    "today we", "today i", "last time", "last lecture", "the plan",
    "we will cover", "we'll cover", "in this lecture", "let's start",
    "let us start", "let's begin", "getting started", "picking up where",
    "on the agenda", "outline for today", "three lectures", "this week we",
]

CLOSING_CUES = [
    "to summarize", "to summarise", "in summary", "to recap", "recap of",
    "next time", "next lecture", "the key takeaway", "takeaways",
    "what we covered", "we covered", "that's all", "that is all",
    "see you", "questions before", "wrap up", "wrapping up",
    "for next class", "read chapter", "homework is",
]

# Trimmed for the job: frequent function words plus the discourse tokens that
# would otherwise dominate any "content word" count on spoken English.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can cannot could couldn't
did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't
it it's its itself let's me more most mustn't my myself no nor not of off on
once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves
just now going get got go come came want need make made take took see saw
know knew think thought say said thing things way ways lot lots really quite
maybe perhaps actually basically okay ok right well yeah yes no um uh like
one two three four five six seven eight nine ten first second next last
""".split())

# Tag questions: a question mark on the end of a statement. "..., right?" is a
# verbal tic, not a question put to the room, and counting it as the latter
# put both reference lectures at ~160 questions an hour. Counted as fillers
# instead, which is what they are -- and which also repairs the filler metric,
# since Whisper strips most um/uh before anyone can count those.
TAG_QUESTIONS = [
    "right", "okay", "ok", "yeah", "yes", "no", "correct", "huh",
    "see", "alright", "you see", "isn't it", "does that make sense",
    "make sense", "got it", "clear", "fine", "agreed", "sure",
]

# Words that are capitalised mid-sentence for reasons other than being a
# person's name. Keeps the PII tripwire from firing on every lecture that
# mentions Python or Monday. Not exhaustive -- the metric is a flag for review.
#
# The second block is what 17-635 lecture 13 threw up in testing: Whisper
# capitalises product and feature nouns mid-sentence freely, so Bedrock, Docs,
# Context and Trip Advisor all read as people. Expect to keep adding to this.
NOT_NAMES = set("""
calendar search weather docs context advisor trip dollar dollars
north south east west lang chain bedrock react node express flask django
sheets slides gmail maps drive meet zoom teams slack notion figma
agent agents tool tools model models prompt prompts token tokens
retrieval embedding embeddings vector index chunk chunks
""".split()) | set("""
monday tuesday wednesday thursday friday saturday sunday january february
march april may june july august september october november december
python java javascript c c++ rust go haskell ocaml sql html css linux unix
windows macos ios android google microsoft apple amazon openai anthropic
github gitlab docker kubernetes aws azure numpy pytorch tensorflow pandas
cmu carnegie mellon pittsburgh america american english european
gpt claude llm api url http https json xml csv pdf gpu cpu ram
fibonacci newton euler gauss turing dijkstra huffman bayes markov nash
boolean gaussian euclidean pythagorean cartesian
i i'm i'll i've i'd ok okay
""".split())
