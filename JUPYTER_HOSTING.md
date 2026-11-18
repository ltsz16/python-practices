# Hosting Jupyter Notebooks on GitHub Pages

Great question! While GitHub Pages doesn't natively **execute** Jupyter notebooks, there are several excellent ways to host and display them:

## Option 1: **nbconvert (Recommended for GitHub Pages)**
Convert Jupyter notebooks to static HTML and commit the HTML files.

```bash
# Install nbconvert
pip install nbconvert

# Convert .ipynb to .html
jupyter nbconvert --to html your_notebook.ipynb

# Or convert with a custom template
jupyter nbconvert --to html --template classic your_notebook.ipynb
```

**Advantages:**
- ✅ Works directly on GitHub Pages
- ✅ Fast loading (static HTML)
- ✅ No execution required
- ✅ All outputs are pre-computed
- ✅ Beautiful formatted code and markdown

**Disadvantages:**
- ❌ Users cannot run/modify code
- ❌ Requires manual updates when notebook changes

## Option 2: **GitHub's Built-in Notebook Viewer**
Push `.ipynb` files to your repository—GitHub automatically renders them!

**Advantages:**
- ✅ No conversion needed
- ✅ Shows code, markdown, and output beautifully
- ✅ Works in any GitHub repository

**Disadvantages:**
- ❌ Only works on github.com (not GitHub Pages static site)
- ❌ Users still cannot execute code

**How to use:**
Just link to your notebook in your GitHub Pages site:
```html
<a href="https://github.com/yourusername/repo/blob/main/notebook.ipynb">
  View Notebook on GitHub
</a>
```

## Option 3: **nbviewer (Free Service)**
Use Jupyter's free `nbviewer` service to display notebooks from any URL.

```
https://nbviewer.org/github/yourusername/yourrepo/blob/main/notebook.ipynb
```

**Advantages:**
- ✅ Works with static GitHub Pages
- ✅ Beautiful rendering
- ✅ No setup needed
- ✅ Free service by Jupyter project

**Disadvantages:**
- ❌ Depends on external service
- ❌ Users cannot execute code

**How to use:**
Link in your GitHub Pages HTML:
```html
<a href="https://nbviewer.org/github/ltsz16/python-practices/blob/main/ed/CIS532%20prod%20distribution/Baby_Name_Classifier.ipynb">
  View Baby Name Classifier
</a>
```

## Option 4: **Binder (Full Execution)**
Deploy notebooks with full Python execution capability using Binder.

**Advantages:**
- ✅ Users can run and modify code
- ✅ Full interactive environment
- ✅ Works with GitHub repositories
- ✅ Free (though limited resources)

**Disadvantages:**
- ❌ Slower startup (~30 seconds to launch environment)
- ⚠️ Limited computational resources

**How to use:**
1. Add a `requirements.txt` file to your repository with dependencies
2. Create a Binder link:
```
https://mybinder.org/v2/gh/ltsz16/python-practices/main?filepath=ed/path/to/notebook.ipynb
```

## Option 5: **Google Colab Links**
Upload notebooks to Google Drive and link from GitHub Pages.

**Advantages:**
- ✅ Full execution environment
- ✅ Free Google account required
- ✅ Modern interface
- ✅ GPU/TPU available (free tier)

**Disadvantages:**
- ❌ Requires separate Google Drive setup
- ❌ Less integrated with GitHub workflow

## Recommendation for Your Repository

Given your structure with educational notebooks in `ed/`, I recommend:

**Primary Approach: nbconvert + GitHub Pages**
1. Convert each `.ipynb` to `.html` using nbconvert
2. Store HTML files in your `ed/` directory
3. Update your links in `ed/README.md` to point to the HTML files
4. Users see formatted, fully-rendered content immediately

**Secondary Approach: Link to nbviewer**
Add links in your markdown that point to nbviewer for quick viewing without conversion.

**Example implementation:**

```markdown
## Learning Resources

### Data Science
- [Baby Name Classifier](https://nbviewer.org/github/ltsz16/python-practices/blob/main/ed/CIS532%20prod%20distribution/Baby_Name_Classifier.ipynb) - Interactive viewer
- [Linear Regression Demo](linear_regression_demo.html) - Pre-rendered HTML

### Linear Classifiers  
- [Perceptron Classifier](https://nbviewer.org/github/ltsz16/python-practices/blob/main/ed/CIS533%20linear%20classifiers/Perceptron_Classifier.ipynb)
```

## Quick Setup Command

Convert all notebooks in your `ed/` directory to HTML:

```bash
cd ed/
find . -name "*.ipynb" -exec jupyter nbconvert --to html {} \;
```

This will create `.html` files next to each `.ipynb` file, viewable on GitHub Pages!
