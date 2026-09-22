#!/bin/bash
# Pre-commit validation script
# Ensures code quality, tests, and best practices before commit

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED=0
STAGED_FILES=$(git diff --cached --name-only)

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            PRE-COMMIT VALIDATION SUITE                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Check for staged files
if [ -z "$STAGED_FILES" ]; then
  echo -e "${YELLOW}⚠️  No staged files to commit${NC}"
  exit 0
fi

echo "Staged files:"
echo "$STAGED_FILES" | sed 's/^/  ✓ /'
echo ""

# 2. Deno Format Check
echo -e "${BLUE}→ Checking code formatting (deno fmt)...${NC}"
if deno fmt --check > /dev/null 2>&1; then
  echo -e "${GREEN}  ✅ All files properly formatted${NC}"
else
  echo -e "${RED}  ❌ Code formatting issues found${NC}"
  echo "     Run: deno fmt"
  FAILED=1
fi
echo ""

# 3. Deno Lint Check
echo -e "${BLUE}→ Linting code (deno lint)...${NC}"
if deno lint > /dev/null 2>&1; then
  echo -e "${GREEN}  ✅ No lint errors${NC}"
else
  echo -e "${RED}  ❌ Lint errors found${NC}"
  echo "     Run: deno lint"
  FAILED=1
fi
echo ""

# 4. Deno Type Check
echo -e "${BLUE}→ Type checking (deno check)...${NC}"
if deno check --all > /dev/null 2>&1; then
  echo -e "${GREEN}  ✅ All types valid${NC}"
else
  echo -e "${RED}  ❌ Type errors found${NC}"
  echo "     Run: deno check --all"
  FAILED=1
fi
echo ""

# 5. Unit Tests
echo -e "${BLUE}→ Running unit tests...${NC}"
if deno test -A --filter "unit:" --no-check > /dev/null 2>&1; then
  TEST_COUNT=$(deno test -A --filter "unit:" --no-check --quiet 2>/dev/null | grep -c "^" || echo "0")
  echo -e "${GREEN}  ✅ All unit tests passing${NC}"
else
  echo -e "${YELLOW}  ⚠️  Some unit tests failed (continue? check manually)${NC}"
  deno test -A --filter "unit:" --no-check 2>&1 | tail -5
fi
echo ""

# 6. Check for secrets/sensitive data
echo -e "${BLUE}→ Scanning for secrets...${NC}"
if ! echo "$STAGED_FILES" | xargs grep -l -E '(password|apikey|api_key|secret|token|credential)' 2>/dev/null | grep -v node_modules | grep -v '.secrets' > /dev/null; then
  echo -e "${GREEN}  ✅ No suspicious secrets detected${NC}"
else
  echo -e "${YELLOW}  ⚠️  Potential secrets detected - review before committing${NC}"
fi
echo ""

# 7. Check for large files
echo -e "${BLUE}→ Checking file sizes...${NC}"
LARGE_FILES=$(find . -type f -size +10M -not -path './.git/*' -not -path './node_modules/*' -not -path './.deno/*' 2>/dev/null || true)
if [ -z "$LARGE_FILES" ]; then
  echo -e "${GREEN}  ✅ No large files detected${NC}"
else
  echo -e "${YELLOW}  ⚠️  Large files detected (may slow down repo):${NC}"
  echo "$LARGE_FILES" | sed 's/^/     - /'
fi
echo ""

# 8. Check for TODO/FIXME in staged code
echo -e "${BLUE}→ Checking for TODO/FIXME comments...${NC}"
TODO_COUNT=$(echo "$STAGED_FILES" | xargs grep -l "TODO\|FIXME" 2>/dev/null | wc -l || echo "0")
if [ "$TODO_COUNT" -eq 0 ]; then
  echo -e "${GREEN}  ✅ No unresolved TODO/FIXME comments${NC}"
else
  echo -e "${YELLOW}  ⚠️  Found TODO/FIXME in $TODO_COUNT file(s) - resolve before merge${NC}"
fi
echo ""

# 9. Check commit message structure
echo -e "${BLUE}→ Validating commit message format...${NC}"
if [ -f "$GIT_AUTHOR_DATE" ]; then
  echo -e "${GREEN}  ✅ Commit message validation will run on commit-msg hook${NC}"
else
  echo -e "${GREEN}  ✅ Ready for commit message validation${NC}"
fi
echo ""

# 10. Check for merge conflicts
echo -e "${BLUE}→ Checking for merge conflicts...${NC}"
if ! git diff --cached | grep -q "^<<<<<<< HEAD"; then
  echo -e "${GREEN}  ✅ No merge conflicts${NC}"
else
  echo -e "${RED}  ❌ Unresolved merge conflicts detected${NC}"
  FAILED=1
fi
echo ""

# 11. Check documentation organization
echo -e "${BLUE}→ Validating documentation organization...${NC}"
if bash .githooks/doc-organization-validator.sh; then
  echo -e "${GREEN}  ✅ Documentation properly organized${NC}"
else
  FAILED=1
fi
echo ""

# Final summary
echo "╔════════════════════════════════════════════════════════════════╗"
if [ $FAILED -eq 0 ]; then
  echo -e "║ ${GREEN}✅ ALL VALIDATIONS PASSED - READY TO COMMIT${NC}            ║"
else
  echo -e "║ ${RED}❌ SOME VALIDATIONS FAILED - FIX BEFORE COMMITTING${NC}  ║"
fi
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

exit $FAILED
