# Refresh status

The scheduled Scholar workflow writes `last-refresh.json` after every attempt.
The status record contains counts, timestamps, the resulting Hugging Face commit
when available, and only the error type on failure. It never stores credentials
or raw exception messages.

Committing this operational record makes failures visible and provides genuine
repository activity so GitHub does not silently disable the monthly schedule.
