#!/bin/bash

# Documentation Organization Validator
# Ensures all .md files are in correct folders and prevents orphan docs

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED=0
WARNINGS=0

# Allowed root-level docs (minimal - only README.md)
# All other .md files must be in docs/ subfolders
ALLOWED_ROOT=(
  "README.md"
)

# Allowed doc directories
ALLOWED_DOC_DIRS=(
  "docs"
  "engine"
  "src"
  "tests"
  ".github"
)

echo -e "${BLUE}→ Validating documentation organization...${NC}"
echo ""

# Find all .md files (exclude .git, node_modules, .tmp, .venv, .direnv)
MARKDOWN_FILES=$(find . \
  -name "*.md" \
  -type f \
  -not -path "./.git/*" \
  -not -path "./node_modules/*" \
  -not -path "./.tmp/*" \
  -not -path "./.venv/*" \
  -not -path "./.direnv/*" \
  -not -path "./.idea/*" \
  | sort)

# Validate each file
while IFS= read -r file; do
  # Remove leading ./
  file="${file#./}"
  
  # Skip hidden files
  if [[ "$file" =~ ^\..*\.md$ ]]; then
    continue
  fi
  
  # Check if at root level (no slashes)
  if [[ ! "$file" =~ / ]]; then
    # Root level - check if in allowed list
    filename=$(basename "$file")
    found=0
    
    for allowed in "${ALLOWED_ROOT[@]}"; do
      if [[ "$filename" == "$allowed" ]]; then
        found=1
        break
      fi
    done
    
    if [[ $found -eq 0 ]]; then
      echo -e "${RED}  ❌ Orphan root doc: ${file}${NC}"
      echo "     Move to: docs/ or appropriate subfolder"
      FAILED=1
    fi
  else
    # In subfolder - verify it's in allowed directory
    first_dir=$(echo "$file" | cut -d/ -f1)
    found=0
    
    for allowed_dir in "${ALLOWED_DOC_DIRS[@]}"; do
      if [[ "$first_dir" == "$allowed_dir" ]]; then
        found=1
        break
      fi
    done
    
    if [[ $found -eq 0 ]]; then
      echo -e "${RED}  ❌ Doc in unexpected location: ${file}${NC}"
      echo "     Move to: docs/, engine/, src/, tests/, or .github/"
      FAILED=1
    fi
  fi
done <<< "$MARKDOWN_FILES"

# Check for common orphan patterns
echo ""
echo -e "${BLUE}→ Checking for orphan documentation patterns...${NC}"

# Check for docs in code directories (common mistake)
if find src -name "*.md" -type f 2>/dev/null | grep -q .; then
  echo -e "${YELLOW}  ⚠️  Found .md files in src/ - consider moving to docs/${NC}"
  WARNINGS=$((WARNINGS + 1))
fi

if find lib -name "*.md" -type f 2>/dev/null | grep -q .; then
  echo -e "${YELLOW}  ⚠️  Found .md files in lib/ - consider moving to docs/${NC}"
  WARNINGS=$((WARNINGS + 1))
fi

# Verify docs/ has proper structure
echo ""
echo -e "${BLUE}→ Validating docs/ folder structure...${NC}"

# Check if docs/README.md exists (index)
if [[ ! -f "docs/README.md" ]]; then
  echo -e "${YELLOW}  ⚠️  Missing docs/README.md (documentation index)${NC}"
  WARNINGS=$((WARNINGS + 1))
fi

# Check for orphan .md files directly in docs/ (should be in subfolders)
orphan_count=$(find docs -maxdepth 1 -name "*.md" -type f | wc -l)
if [[ $orphan_count -gt 1 ]]; then
  echo -e "${YELLOW}  ⚠️  Found $orphan_count .md files in docs/ root (prefer subfolders)${NC}"
  find docs -maxdepth 1 -name "*.md" -type f | sed 's/^/     - /'
  WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo ""
if [[ $FAILED -eq 0 ]]; then
  echo -e "${GREEN}  ✅ All documentation properly organized${NC}"
else
  echo -e "${RED}  ❌ Documentation organization violations found${NC}"
fi

if [[ $WARNINGS -gt 0 ]]; then
  echo -e "${YELLOW}  ⚠️  $WARNINGS warnings (best practice)${NC}"
fi

echo ""

exit $FAILED
