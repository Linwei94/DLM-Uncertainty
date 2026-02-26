from default_utils.registry import METRICS_FUNCTIONS, GRADER_FUNCTIONS, CONFIDENCE_FUNCTIONS, PROMPT_FORMATTER
from main import _auto_import_modules
_auto_import_modules()

print("Available confidence metrics:")
for method in CONFIDENCE_FUNCTIONS.keys():
    print(f" - {method}")
print()
print("Available performance metrics:")
for metric in METRICS_FUNCTIONS.keys():
    print(f" - {metric}")
print()
print("Available graders:")
for grader in GRADER_FUNCTIONS.keys():
    print(f" - {grader}")
print()
print("Available prompt formatters:")
for formatter in PROMPT_FORMATTER.keys():
    print(f" - {formatter}")