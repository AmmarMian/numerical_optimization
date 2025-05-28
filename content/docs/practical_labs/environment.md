---
title: Lab environment
weight : 50
---

# Lab Environment Setup

Welcome to the numerical optimization course! This page will guide you through setting up a modern, efficient Python environment using **uv**.

## Prerequisites

**Good news!** uv doesn't require Python to be pre-installed - it can manage Python installations for you. However, having Python already installed won't hurt.

## Installing uv

### 🐧 Linux & 🍎 macOS

The fastest way to install uv is using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This will:
- Download and install the latest version of uv
- Add uv to your PATH automatically
- Work on both Linux and macOS

**Alternative installation methods:**
- **Using pipx** (if you have it): `pipx install uv`
- **Using pip**: `pip install uv`
- **Using Homebrew** (macOS): `brew install uv`

### 🪟 Windows

**Option 1: PowerShell Installer (Recommended)**
Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Option 2: Using pipx or pip**
If you have Python already installed:
```powershell
pipx install uv
# or
pip install uv
```

**Option 3: Download from GitHub**
You can also download the installer or binaries directly from the GitHub releases page.

### Verify Installation

After installation, restart your terminal and verify uv is working:

```bash
uv --version
```

You should see output like `uv 0.7.8` or similar (version numbers may vary).

## Setting Up the Lab Project

Now let's create a dedicated project for all your numerical optimization lab sessions.

### Step 1: Create the Project

Navigate to where you want to store your course materials and run:

```bash
uv init --bare numerical-optimization-labs
cd numerical-optimization-labs
```

The `--bare` flag creates a minimal project structure with only essential files: `pyproject.toml`, `.python-version`, and `README.md` (no default `main.py` file).

### Step 2: Install Required Packages

Now we'll install all the packages you'll need for the course. The `uv add` command will automatically create a virtual environment, install packages, and update both `pyproject.toml` and the lock file:

```bash
# Core scientific computing packages
uv add numpy scipy scikit-learn

# Visualization libraries
uv add matplotlib plotly

# Machine learning framework
uv add torch

# Development and interactive tools
uv add rich jupyter ipython

# Optional: Add some useful development tools
uv add pytest black ruff
```

**What's happening behind the scenes:**
- uv creates a virtual environment at `.venv/` in your project directory
- All packages are installed into this isolated environment
- A lockfile (`uv.lock`) is generated containing exact versions of all dependencies for reproducible installations
- Your `pyproject.toml` is updated with the new dependencies

### Step 3: Verify Installation

Check that everything installed correctly:

```bash
uv run python -c "import numpy, scipy, sklearn, matplotlib, plotly, torch, rich, jupyter, IPython; print('✅ All packages imported successfully!')"
```

## Using Your Lab Environment

### Running Python Scripts

To run any Python script in your lab environment:

```bash
uv run python your_script.py
```

The `uv run` command ensures your script runs in the project's virtual environment with all dependencies available.

### Starting Jupyter Lab/Notebook

To start Jupyter for interactive development:

```bash
# For Jupyter Lab (recommended)
uv run jupyter lab

# For classic Jupyter Notebook
uv run jupyter notebook
```

### Interactive Python (IPython)

For an enhanced interactive Python experience:

```bash
uv run ipython
```

### Adding More Packages Later

If you need additional packages during the course:

```bash
uv add package-name
```

To remove packages you no longer need:

```bash
uv remove package-name
```

## Project Structure

Your lab project will look like this:

```
numerical-optimization-labs/
├── .venv/                 # Virtual environment (auto-created)
├── .python-version        # Pinned Python version
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Exact dependency versions (for reproducibility)
├── README.md             # Project documentation
└── your_lab_files.py     # Your lab work goes here
```

## Sharing and Collaboration

### Setting up the Environment on Another Machine

If you clone this project or share it with others, they can recreate the exact environment by running:

```bash
cd numerical-optimization-labs
uv sync
```

This command reads the lockfile and installs the exact same versions of all dependencies.

### Version Control

Make sure to commit these files to Git:
- ✅ `pyproject.toml`
- ✅ `uv.lock`
- ✅ `.python-version`
- ❌ `.venv/` (add this to `.gitignore`)

## Troubleshooting

### Permission Issues
If you encounter permission errors, use `sudo` on macOS/Linux or run your command prompt as administrator on Windows.

### Python Version Issues
If you need a specific Python version:

```bash
# Install a specific Python version
uv python install 3.11

# Pin it to your project
uv python pin 3.11
```

### Environment Issues
If something goes wrong with your environment:

```bash
# Sync environment with lockfile
uv sync

# Force recreate environment
rm -rf .venv
uv sync
```

### Package Conflicts
uv's dependency resolver is much more robust than pip and should handle conflicts automatically. If you encounter issues, check the error message and try updating conflicting packages.

## Quick Reference Commands

| Task | Command |
|------|---------|
| Create new project | `uv init project-name` |
| Add packages | `uv add package1 package2` |
| Remove packages | `uv remove package-name` |
| Run Python script | `uv run python script.py` |
| Start Jupyter | `uv run jupyter lab` |
| Install from lockfile | `uv sync` |
| List installed packages | `uv tree` |
| Update a package | `uv add package-name --upgrade` |

## Getting Help

- **uv Documentation**: [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **Command Help**: `uv help` or `uv command --help`
- **Course Forum**: [link to your course forum/discussion board]

---

🎉 **You're all set!** Your lab environment is ready for numerical optimization adventures. If you encounter any issues during setup, don't hesitate to ask for help during class or by mail.
