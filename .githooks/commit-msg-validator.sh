#!/bin/bash
# Commit message validation script
# Ensures commits follow best practices and conventions

COMMIT_MSG_FILE="$1"
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")
COMMIT_SOURCE="$2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Exit early for merge/revert commits
if [ "$COMMIT_SOURCE" = "merge" ] || [ "$COMMIT_SOURCE" = "squash" ] || echo "$COMMIT_MSG" | grep -q "^Merge\|^Revert"; then
  exit 0
fi

ERRORS=()
WARNINGS=()

# 1. Check commit message is not empty
if [ -z "$(echo "$COMMIT_MSG" | grep -v '^#')" ]; then
  ERRORS+=("Commit message is empty")
fi

# 2. Check first line format (conventional commit)
FIRST_LINE=$(echo "$COMMIT_MSG" | head -1)
if ! echo "$FIRST_LINE" | grep -qE '^(feat|fix|docs|style|refactor|perf|test|chore|ci|build|revert)(\(.+\))?!?:'; then
  WARNINGS+=("First line should follow conventional commit format: <type>(<scope>): <subject>")
fi

# 3. Check first line length (max 72 chars)
if [ ${#FIRST_LINE} -gt 72 ]; then
  WARNINGS+=("First line should be max 72 characters (got ${#FIRST_LINE})")
fi

# 4. Check first line capitalization
if ! echo "$FIRST_LINE" | grep -qE '^[a-z]+.*: [A-Z]'; then
  WARNINGS+=("Subject should start with capital letter after colon")
fi

# 5. Check for issue/ticket reference
if ! echo "$COMMIT_MSG" | grep -qiE '(#[0-9]+|JIRA|closes|fixes|relates to)'; then
  WARNINGS+=("Consider adding issue reference (e.g., #123 or closes #456)")
fi

# 6. Check body separation (blank line after first line)
LINE_COUNT=$(echo "$COMMIT_MSG" | wc -l)
if [ $LINE_COUNT -gt 1 ]; then
  SECOND_LINE=$(echo "$COMMIT_MSG" | sed -n '2p')
  if [ ! -z "$SECOND_LINE" ] && [ "$SECOND_LINE" != "#"* ]; then
    WARNINGS+=("Add blank line between subject and body")
  fi
fi

# 7. Check for imperative mood in subject
SUBJECT=$(echo "$FIRST_LINE" | cut -d':' -f2- | xargs)
if echo "$SUBJECT" | grep -qiE '^(added|removed|fixed|changed|updated)'; then
  WARNINGS+=("Use imperative mood in subject (e.g., 'Add' instead of 'Added')")
fi

# 8. Check line length in body (max 100 chars)
BODY=$(echo "$COMMIT_MSG" | tail -n +3)
while IFS= read -r line; do
  if [ ! -z "$line" ] && ! echo "$line" | grep -q "^#"; then
    if [ ${#line} -gt 100 ]; then
      WARNINGS+=("Body line too long (${#line} chars): ${line:0:50}...")
    fi
  fi
done <<< "$BODY"

# 9. Check for TODO or FIXME in commit message
if echo "$COMMIT_MSG" | grep -qi "TODO\|FIXME"; then
  WARNINGS+=("Remove TODO/FIXME comments from commit message")
fi

# 10. Check for debugging code indicators
if echo "$COMMIT_MSG" | grep -qiE 'console\.log|debugger|var dump'; then
  ERRORS+=("Commit message mentions debug code - ensure it's removed")
fi

# Print validation results
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "   COMMIT MESSAGE VALIDATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Message Preview:"
echo "  $(echo "$FIRST_LINE" | head -c 70)..."
echo ""

if [ ${#ERRORS[@]} -gt 0 ]; then
  echo -e "${RED}❌ ERRORS (must fix):${NC}"
  for error in "${ERRORS[@]}"; do
    echo -e "   ${RED}✗${NC} $error"
  done
  echo ""
fi

if [ ${#WARNINGS[@]} -gt 0 ]; then
  echo -e "${YELLOW}⚠️  WARNINGS (best practice):${NC}"
  for warning in "${WARNINGS[@]}"; do
    echo -e "   ${YELLOW}→${NC} $warning"
  done
  echo ""
fi

if [ ${#ERRORS[@]} -eq 0 ] && [ ${#WARNINGS[@]} -eq 0 ]; then
  echo -e "${GREEN}✅ COMMIT MESSAGE VALID${NC}"
  echo ""
fi

echo "═══════════════════════════════════════════════════════════════"
echo ""

# Exit with error if there are critical errors
if [ ${#ERRORS[@]} -gt 0 ]; then
  exit 1
fi

exit 0
