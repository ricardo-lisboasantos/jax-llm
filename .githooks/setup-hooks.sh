#!/bin/bash
# Setup pre-commit and commit-msg hooks for the project

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.githooks"
GIT_HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Setting up Git hooks..."
echo ""

# Create .git/hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Function to install hook
install_hook() {
  local hook_name=$1
  local hook_file=$2
  local git_hook="$GIT_HOOKS_DIR/$hook_name"
  
  echo -n "Installing $hook_name hook... "
  
  # Create wrapper script
  cat > "$git_hook" << HOOK_SCRIPT
#!/bin/bash
# Auto-generated hook - do not edit
exec "$HOOKS_DIR/$hook_file" "\$@"
HOOK_SCRIPT
  
  chmod +x "$git_hook"
  echo "✅"
}

# Install hooks
install_hook "pre-commit" "pre-commit-validator.sh"
install_hook "commit-msg" "commit-msg-validator.sh"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Git hooks installed successfully!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Installed hooks:"
echo "  • pre-commit   → Code quality, tests, formatting"
echo "  • commit-msg   → Conventional commit validation"
echo ""
echo "Next time you commit, these validations will run automatically."
echo "To bypass hooks (not recommended): git commit --no-verify"
echo ""
