"""
Deterministic HTML exporter for CourseState.
Converts the structured JSON data into a complete HTML page with interactive elements.
"""

from urllib.parse import urlparse

from workflows.state import ActivitiesSection, CourseState, HtmlElement, MetaElements, Section


def is_valid_image_url(url: str) -> bool:
    """
    Quick validation that a URL or local path is suitable for embedding in HTML.

    Args:
        url: The image URL or local file path to validate

    Returns:
        True if the URL/path appears valid for HTML embedding
    """
    if not url or not isinstance(url, str):
        return False

    url = url.strip()

    # Accept local file paths (relative or absolute) pointing to image files
    if not url.startswith(("http://", "https://")):
        import os

        return os.path.isfile(url) or url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"))

    # URL shouldn't be too long
    if len(url) > 2000:
        return False

    # Basic URL structure check
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False
    except Exception:
        return False

    return True


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not isinstance(text, str):
        return str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_element(element: HtmlElement) -> str:
    """Render a single HTML element based on its type."""
    if element.type == "p":
        return f"<p>{escape_html(element.content)}</p>"

    elif element.type == "ul":
        if isinstance(element.content, list):
            items = "".join(f"<li>{escape_html(str(item))}</li>" for item in element.content)
            return f"<ul>{items}</ul>"
        return ""

    elif element.type == "quote":
        quote_data = element.content
        if isinstance(quote_data, dict):
            quote_text = escape_html(str(quote_data.get("text", "")))
            author = escape_html(str(quote_data.get("author", "")))
            return f'<blockquote class="quote"><p>{quote_text}</p><footer>— {author}</footer></blockquote>'
        return ""

    elif element.type == "table":
        table_data = element.content
        if isinstance(table_data, dict):
            title = escape_html(str(table_data.get("title", "")))
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])

            html = f'<div class="table-container"><h4>{title}</h4><table>'

            # Headers
            if headers:
                html += "<thead><tr>"
                for header in headers:
                    html += f"<th>{escape_html(str(header))}</th>"
                html += "</tr></thead>"

            # Rows
            if rows:
                html += "<tbody>"
                for row in rows:
                    html += "<tr>"
                    for cell in row:
                        html += f"<td>{escape_html(str(cell))}</td>"
                    html += "</tr>"
                html += "</tbody>"

            html += "</table></div>"
            return html
        return ""

    elif element.type == "paragraphs":
        # Render nested blocks
        html = '<div class="content-items">'
        if isinstance(element.content, list):
            for block in element.content:
                html += '<div class="content-item">'
                html += f'<h4 class="item-title"><i class="{block.icon}"></i> {escape_html(block.title)}</h4>'
                html += '<div class="item-body">'
                # Render image if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div></div>"
        html += "</div>"
        return html

    elif element.type == "accordion":
        # Render accordion with collapsible sections
        html = '<div class="accordion-container">'
        if isinstance(element.content, list):
            for idx, block in enumerate(element.content):
                html += '<div class="accordion-item">'
                html += '<div class="accordion-header" onclick="toggleAccordion(this)">'
                html += f'<i class="{block.icon}"></i> {escape_html(block.title)}'
                html += '<span class="accordion-toggle">▼</span>'
                html += "</div>"
                html += f'<div class="accordion-body" style="display: {"block" if idx == 0 else "none"};">'
                # Render image if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div></div>"
        html += "</div>"
        return html

    elif element.type == "tabs":
        # Render tabs interface
        html = '<div class="tabs-container">'
        if isinstance(element.content, list):
            # Render tab headers
            html += '<div class="tabs-header">'
            for idx, block in enumerate(element.content):
                active_class = " active" if idx == 0 else ""
                html += f'<button class="tab-button{active_class}" onclick="switchTab(this, \'tab-{idx}\')">'
                html += f'<i class="{block.icon}"></i> {escape_html(block.title)}'
                html += "</button>"
            html += "</div>"
            # Render tab content
            for idx, block in enumerate(element.content):
                display_style = "block" if idx == 0 else "none"
                html += f'<div class="tab-content" id="tab-{idx}" style="display: {display_style};">'
                # Render image if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div>"
        html += "</div>"
        return html

    elif element.type == "carousel":
        # Render carousel/slideshow
        html = '<div class="carousel-container">'
        if isinstance(element.content, list):
            for idx, block in enumerate(element.content):
                display_style = "block" if idx == 0 else "none"
                html += f'<div class="carousel-slide" style="display: {display_style};">'
                html += f'<h4><i class="{block.icon}"></i> {escape_html(block.title)}</h4>'
                # Render image if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div>"
            html += '<div class="carousel-controls">'
            html += '<button onclick="prevSlide(this)">❮ Previous</button>'
            html += '<button onclick="nextSlide(this)">Next ❯</button>'
            html += "</div>"
        html += "</div>"
        return html

    elif element.type == "flip":
        # Render flip cards
        html = '<div class="flip-container">'
        if isinstance(element.content, list):
            for block in element.content:
                html += '<div class="flip-card" onclick="this.classList.toggle(\'flipped\')">'
                html += '<div class="flip-card-inner">'
                html += '<div class="flip-card-front">'
                # Render image on front if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image flip-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                html += f'<h4><i class="{block.icon}"></i> {escape_html(block.title)}</h4>'
                html += '<p class="flip-hint">Click to flip</p>'
                html += "</div>"
                html += '<div class="flip-card-back">'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div>"
                html += "</div></div>"
        html += "</div>"
        return html

    elif element.type == "timeline":
        # Render timeline view
        html = '<div class="timeline-container">'
        if isinstance(element.content, list):
            for idx, block in enumerate(element.content):
                side_class = "left" if idx % 2 == 0 else "right"
                html += f'<div class="timeline-item {side_class}">'
                html += '<div class="timeline-marker"></div>'
                html += '<div class="timeline-content">'
                html += f'<h4><i class="{block.icon}"></i> {escape_html(block.title)}</h4>'
                # Render image if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div></div>"
        html += "</div>"
        return html

    elif element.type == "conversation":
        # Render conversation/dialog style
        html = '<div class="conversation-container">'
        if isinstance(element.content, list):
            for idx, block in enumerate(element.content):
                side_class = "left" if idx % 2 == 0 else "right"
                html += f'<div class="conversation-message {side_class}">'
                html += '<div class="message-bubble">'
                html += f'<div class="message-header"><i class="{block.icon}"></i> {escape_html(block.title)}</div>'
                html += '<div class="message-body">'
                # Render image if present and valid
                if hasattr(block, "image") and block.image and block.image.get("content"):
                    img_url = block.image.get("content", "")
                    if is_valid_image_url(img_url):
                        html += f'<div class="block-image"><img src="{escape_html(img_url)}" alt="{escape_html(block.title)}" loading="lazy" onerror="this.style.display=\'none\'"></div>'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += "</div></div></div>"
        html += "</div>"
        return html

    return ""


def render_meta_elements(meta: MetaElements) -> str:
    """Render metadata elements: glossary, key concept, interesting fact, quote."""
    html = '<div class="meta-elements-section">'

    # Key Concept
    if meta.key_concept:
        html += '<div class="key-concept-box">'
        html += '<h4 class="key-concept-title"><i class="mdi mdi-lightbulb-on"></i> Concepto Clave</h4>'
        html += f"<p>{escape_html(meta.key_concept)}</p>"
        html += "</div>"

    # Interesting Fact
    if meta.interesting_fact:
        html += '<div class="interesting-fact-box">'
        html += '<h4 class="fact-title"><i class="mdi mdi-brain"></i> Dato Interesante</h4>'
        html += f"<p>{escape_html(meta.interesting_fact)}</p>"
        html += "</div>"

    # Glossary
    if meta.glossary:
        html += '<div class="glossary-section">'
        html += '<h4 class="glossary-title"><i class="mdi mdi-book-alphabet"></i> Glosario</h4>'
        html += '<dl class="glossary-list">'
        for term in meta.glossary:
            html += f'<dt class="glossary-term">{escape_html(term.term)}</dt>'
            html += f'<dd class="glossary-explanation">{escape_html(term.explanation)}</dd>'
        html += "</dl>"
        html += "</div>"

    # Quote
    if meta.quote:
        quote_text = escape_html(str(meta.quote.get("text", "")))
        author = escape_html(str(meta.quote.get("author", "")))
        html += f'<blockquote class="meta-quote"><p>{quote_text}</p><footer>— {author}</footer></blockquote>'

    html += "</div>"
    return html


def render_activities_section(activities: ActivitiesSection) -> str:
    """Render activities section with quiz and application activities."""
    html = '<div class="activities-section">'
    html += '<h4 class="activities-title"><i class="mdi mdi-puzzle"></i> Actividades</h4>'

    # Quiz Activities
    if activities.quiz:
        html += '<div class="quiz-activities">'
        html += '<h5 class="quiz-subtitle">Evaluación</h5>'
        for idx, activity in enumerate(activities.quiz, 1):
            activity_type_map = {
                "order_list": ("Ordenar Lista", "mdi-order-numeric-ascending"),
                "fill_gaps": ("Completar Espacios", "mdi-form-textbox"),
                "swipper": ("Verdadero/Falso", "mdi-swap-horizontal"),
                "linking_terms": ("Relacionar Términos", "mdi-link-variant"),
                "multiple_choice": ("Opción Múltiple", "mdi-checkbox-multiple-marked"),
                "multi_selection": ("Selección Múltiple", "mdi-checkbox-marked-circle"),
            }
            type_label, icon = activity_type_map.get(activity.type, ("Actividad", "mdi-check"))

            html += '<div class="activity-card quiz-card" onclick="toggleActivity(this)">'
            html += '<div class="activity-header">'
            html += f'<i class="{icon}"></i> <strong>{type_label} {idx}</strong>'
            html += '<span class="activity-toggle">▼</span>'
            html += "</div>"
            html += '<div class="activity-body" style="display: none;">'

            # Render activity content based on type
            content = activity.content
            if hasattr(content, "question"):
                html += f'<p class="activity-question">{escape_html(content.question)}</p>'

            html += "</div></div>"
        html += "</div>"

    # Application Activities
    if activities.application:
        html += '<div class="application-activities">'
        html += '<h5 class="application-subtitle">Aplicación Práctica</h5>'
        for _, activity in enumerate(activities.application, 1):
            activity_type_map = {
                "group_activity": ("Actividad Grupal", "mdi-account-group"),
                "discussion_forum": ("Foro de Discusión", "mdi-forum"),
                "individual_project": ("Proyecto Individual", "mdi-account-edit"),
                "open_ended_quiz": ("Pregunta Abierta", "mdi-comment-question"),
            }
            type_label, icon = activity_type_map.get(activity.type, ("Actividad", "mdi-check"))

            html += '<div class="activity-card application-card">'
            html += '<div class="activity-header-static">'
            html += f'<i class="{icon}"></i> <strong>{type_label}</strong>'
            html += "</div>"
            html += '<div class="activity-content">'
            if hasattr(activity.content, "question"):
                html += f"<p>{escape_html(activity.content.question)}</p>"
            html += "</div></div>"
        html += "</div>"

    html += "</div>"
    return html


def render_section(section: Section, section_num: int) -> str:
    """Render a complete section with all its content in a simple, linear format."""
    html = f'<section class="course-section" id="section-{section_num}">'
    html += f'<h3 class="section-title">{escape_html(section.title)}</h3>'

    # HTML Structure - now a direct array
    if section.html:
        # Iterate through html elements
        for element in section.html:
            # Wrap intro in specific class if it's the first p
            if element == section.html[0] and element.type == "p":
                html += '<div class="section-intro">'
                html += render_element(element)
                html += "</div>"
            # Wrap conclusion in specific class if it's the last p
            elif element == section.html[-1] and element.type == "p":
                html += '<div class="section-conclusion">'
                html += render_element(element)
                html += "</div>"
            else:
                html += render_element(element)

    # Render meta elements (glossary, key concepts, facts, quotes)
    if section.meta_elements:
        html += render_meta_elements(section.meta_elements)

    # Render activities (quiz and application)
    if section.activities:
        html += render_activities_section(section.activities)

    html += "</section>"
    return html


def _make_image_path_relative(img_path: str, html_dir: str) -> str:
    """Convert an absolute or project-relative image path to be relative to the HTML file."""
    import os

    if not img_path or img_path.startswith(("http://", "https://")):
        return img_path
    abs_img = os.path.abspath(img_path)
    abs_html_dir = os.path.abspath(html_dir)
    try:
        return os.path.relpath(abs_img, abs_html_dir)
    except ValueError:
        return img_path


def export_to_html(course_state: CourseState, output_path: str) -> None:
    """
    Export the complete course to a clean, readable HTML file.

    Args:
        course_state: The complete course state with all content
        output_path: Path where the HTML file should be saved
    """
    html = f"""<!DOCTYPE html>
<html lang="{course_state.language.lower()[:2]}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape_html(course_state.title)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@mdi/font@7.4.47/css/materialdesignicons.min.css">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.8;
            color: #2c3e50;
            background: #fafafa;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
        }}

        /* Header */
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
            margin: -20px -20px 40px -20px;
        }}
        h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 300;
            letter-spacing: 1px;
        }}
        .course-meta {{
            font-size: 1.1em;
            opacity: 0.95;
            margin: 10px 0;
        }}

        /* Module Structure */
        .module {{
            margin-bottom: 50px;
            page-break-inside: avoid;
        }}
        .module-title {{
            color: #667eea;
            font-size: 2.2em;
            margin: 40px 0 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
            font-weight: 400;
        }}
        .submodule {{
            margin: 30px 0;
            padding-left: 20px;
            border-left: 3px solid #e0e7ff;
        }}
        .submodule-title {{
            color: #764ba2;
            font-size: 1.8em;
            margin-bottom: 8px;
            font-weight: 400;
        }}
        .submodule-description {{
            color: #555;
            font-size: 1.05em;
            font-style: italic;
            margin-bottom: 20px;
            line-height: 1.5;
        }}

        /* Section */
        .course-section {{
            margin: 30px 0;
            padding: 25px;
            background: #fefefe;
            border-radius: 8px;
            border: 1px solid #e8e8e8;
        }}
        .section-title {{
            color: #2c3e50;
            font-size: 1.6em;
            margin-bottom: 20px;
            font-weight: 500;
        }}

        /* Content */
        .section-intro {{
            background: linear-gradient(to right, #f0f4ff, #fefefe);
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
            font-size: 1.05em;
        }}

        .content-items {{
            margin: 25px 0;
        }}
        .content-item {{
            margin: 25px 0;
            padding: 20px 0;
        }}
        .item-title {{
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }}
        .item-title i {{
            font-size: 1.2em;
        }}
        .item-body {{
            padding-left: 35px;
        }}

        /* Block Images */
        .block-image {{
            margin: 15px 0;
            text-align: center;
        }}
        .block-image img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            object-fit: contain;
        }}
        .flip-image {{
            margin: 10px 0;
        }}
        .flip-image img {{
            max-height: 120px;
        }}

        .section-conclusion {{
            background: linear-gradient(to right, #fefefe, #f0f4ff);
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #764ba2;
            font-style: italic;
        }}

        /* Typography */
        p {{
            margin: 15px 0;
            text-align: justify;
            font-size: 1.05em;
        }}
        ul {{
            margin: 15px 0 15px 25px;
            list-style-type: disc;
        }}
        li {{
            margin: 8px 0;
            font-size: 1.05em;
        }}

        /* Special Elements */
        .quote {{
            background: #fff9e6;
            border-left: 5px solid #f59e0b;
            padding: 20px 25px;
            margin: 25px 0;
            font-style: italic;
            font-size: 1.1em;
            color: #744210;
        }}
        .quote footer {{
            text-align: right;
            margin-top: 15px;
            font-weight: bold;
            font-style: normal;
        }}

        .table-container {{
            margin: 25px 0;
            overflow-x: auto;
        }}
        .table-container h4 {{
            margin-bottom: 10px;
            color: #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}

        /* Table of Contents */
        .table-of-contents {{
            background: #f8f9fa;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 30px;
            margin: 40px 0;
        }}
        .table-of-contents h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .toc-module {{
            margin: 20px 0;
        }}
        .toc-module-title {{
            color: #667eea;
            font-size: 1.3em;
            font-weight: 600;
            margin: 15px 0 10px;
        }}
        .toc-submodule {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        .toc-submodule-title {{
            color: #764ba2;
            font-size: 1.15em;
            font-weight: 500;
            margin: 8px 0 5px;
        }}
        .toc-sections {{
            margin-left: 40px;
            list-style: none;
            padding: 0;
        }}
        .toc-sections li {{
            margin: 5px 0;
        }}
        .toc-sections a {{
            color: #2c3e50;
            text-decoration: none;
            display: block;
            padding: 5px 10px;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .toc-sections a:hover {{
            background: #e0e7ff;
            color: #667eea;
            transform: translateX(5px);
        }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 40px 20px;
            color: #7f8c8d;
            border-top: 2px solid #ecf0f1;
            margin-top: 60px;
            font-size: 0.95em;
        }}

        /* Accordion Styles */
        .accordion-container {{
            margin: 25px 0;
        }}
        .accordion-item {{
            border: 1px solid #e0e7ff;
            border-radius: 8px;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        .accordion-header {{
            background: linear-gradient(to right, #f0f4ff, #fefefe);
            padding: 15px 20px;
            cursor: pointer;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #667eea;
        }}
        .accordion-header:hover {{
            background: linear-gradient(to right, #e0e7ff, #f0f4ff);
        }}
        .accordion-header i {{
            margin-right: 10px;
        }}
        .accordion-body {{
            padding: 20px;
            background: white;
        }}
        .accordion-toggle {{
            transition: transform 0.3s;
        }}

        /* Tabs Styles */
        .tabs-container {{
            margin: 25px 0;
        }}
        .tabs-header {{
            display: flex;
            gap: 5px;
            border-bottom: 2px solid #667eea;
            margin-bottom: 20px;
        }}
        .tab-button {{
            background: #f8f9fa;
            border: none;
            padding: 12px 20px;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            font-size: 1em;
            color: #2c3e50;
            transition: all 0.3s;
        }}
        .tab-button:hover {{
            background: #e0e7ff;
        }}
        .tab-button.active {{
            background: #667eea;
            color: white;
        }}
        .tab-button i {{
            margin-right: 8px;
        }}
        .tab-content {{
            padding: 20px;
            background: #fefefe;
            border-radius: 0 8px 8px 8px;
        }}

        /* Carousel Styles */
        .carousel-container {{
            margin: 25px 0;
            position: relative;
            padding: 20px;
            background: #fefefe;
            border-radius: 8px;
            border: 2px solid #e0e7ff;
        }}
        .carousel-slide {{
            min-height: 200px;
        }}
        .carousel-controls {{
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }}
        .carousel-controls button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
        }}
        .carousel-controls button:hover {{
            background: #764ba2;
        }}

        /* Flip Card Styles */
        .flip-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
        .flip-card {{
            perspective: 1000px;
            height: 250px;
            cursor: pointer;
        }}
        .flip-card-inner {{
            position: relative;
            width: 100%;
            height: 100%;
            transition: transform 0.6s;
            transform-style: preserve-3d;
        }}
        .flip-card.flipped .flip-card-inner {{
            transform: rotateY(180deg);
        }}
        .flip-card-front, .flip-card-back {{
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            border-radius: 8px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        .flip-card-front {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .flip-card-back {{
            background: #fefefe;
            border: 2px solid #667eea;
            transform: rotateY(180deg);
            overflow-y: auto;
        }}
        .flip-hint {{
            margin-top: 10px;
            opacity: 0.8;
            font-size: 0.9em;
        }}

        /* Timeline Styles */
        .timeline-container {{
            position: relative;
            margin: 40px 0;
            padding: 20px 0;
        }}
        .timeline-container::before {{
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 4px;
            background: #667eea;
            transform: translateX(-50%);
        }}
        .timeline-item {{
            position: relative;
            width: 45%;
            margin-bottom: 30px;
        }}
        .timeline-item.left {{
            left: 0;
            text-align: right;
        }}
        .timeline-item.right {{
            left: 55%;
        }}
        .timeline-marker {{
            position: absolute;
            width: 20px;
            height: 20px;
            background: #667eea;
            border: 4px solid white;
            border-radius: 50%;
            top: 0;
        }}
        .timeline-item.left .timeline-marker {{
            right: -62px;
        }}
        .timeline-item.right .timeline-marker {{
            left: -62px;
        }}
        .timeline-content {{
            padding: 20px;
            background: #fefefe;
            border-radius: 8px;
            border: 2px solid #e0e7ff;
        }}

        /* Conversation Styles */
        .conversation-container {{
            margin: 25px 0;
        }}
        .conversation-message {{
            margin-bottom: 20px;
            display: flex;
        }}
        .conversation-message.left {{
            justify-content: flex-start;
        }}
        .conversation-message.right {{
            justify-content: flex-end;
        }}
        .message-bubble {{
            max-width: 70%;
            border-radius: 12px;
            overflow: hidden;
        }}
        .conversation-message.left .message-bubble {{
            background: #f0f4ff;
            border: 2px solid #667eea;
        }}
        .conversation-message.right .message-bubble {{
            background: #fff9e6;
            border: 2px solid #f59e0b;
        }}
        .message-header {{
            padding: 10px 15px;
            font-weight: 600;
            color: #667eea;
            border-bottom: 1px solid #e0e7ff;
        }}
        .message-body {{
            padding: 15px;
        }}

        /* Meta Elements Styles */
        .meta-elements-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .key-concept-box {{
            background: linear-gradient(to right, #e0f7fa, #ffffff);
            border-left: 5px solid #00acc1;
            padding: 20px;
            margin: 15px 0;
            border-radius: 6px;
        }}
        .key-concept-title {{
            color: #00838f;
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .interesting-fact-box {{
            background: linear-gradient(to right, #fff3e0, #ffffff);
            border-left: 5px solid #ff9800;
            padding: 20px;
            margin: 15px 0;
            border-radius: 6px;
        }}
        .fact-title {{
            color: #e65100;
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .glossary-section {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 6px;
            border: 2px solid #e0e7ff;
        }}
        .glossary-title {{
            color: #667eea;
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .glossary-list {{
            margin: 0;
            padding: 0;
        }}
        .glossary-term {{
            font-weight: 600;
            color: #667eea;
            margin-top: 12px;
            font-size: 1.05em;
        }}
        .glossary-explanation {{
            margin: 5px 0 5px 20px;
            color: #2c3e50;
            line-height: 1.6;
        }}

        .meta-quote {{
            background: #fff9e6;
            border-left: 5px solid #f59e0b;
            padding: 20px 25px;
            margin: 20px 0;
            font-style: italic;
            font-size: 1.1em;
            color: #744210;
        }}

        /* Activities Styles */
        .activities-section {{
            margin: 30px 0;
            padding: 25px;
            background: #f0f4ff;
            border-radius: 8px;
            border: 2px solid #667eea;
        }}
        .activities-title {{
            color: #667eea;
            font-size: 1.4em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .quiz-activities, .application-activities {{
            margin: 20px 0;
        }}
        .quiz-subtitle, .application-subtitle {{
            color: #764ba2;
            font-size: 1.2em;
            margin-bottom: 15px;
            font-weight: 500;
        }}

        .activity-card {{
            background: white;
            border-radius: 8px;
            margin: 10px 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: box-shadow 0.3s;
        }}
        .activity-card:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}

        .quiz-card {{
            border-left: 4px solid #667eea;
        }}
        .application-card {{
            border-left: 4px solid #10b981;
        }}

        .activity-header {{
            padding: 15px 20px;
            background: linear-gradient(to right, #f0f4ff, #fefefe);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.05em;
        }}
        .activity-header:hover {{
            background: linear-gradient(to right, #e0e7ff, #f0f4ff);
        }}
        .activity-header i {{
            margin-right: 8px;
            color: #667eea;
        }}

        .activity-header-static {{
            padding: 15px 20px;
            background: linear-gradient(to right, #ecfdf5, #fefefe);
            font-size: 1.05em;
        }}
        .activity-header-static i {{
            margin-right: 8px;
            color: #10b981;
        }}

        .activity-body {{
            padding: 20px;
            border-top: 1px solid #e0e7ff;
        }}
        .activity-content {{
            padding: 15px 20px;
        }}

        .activity-question {{
            font-weight: 500;
            color: #2c3e50;
            margin: 0;
        }}

        .activity-toggle {{
            transition: transform 0.3s;
            color: #667eea;
            font-size: 1.2em;
        }}

        /* Print Styles */
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
            .course-section {{ page-break-inside: avoid; }}
        }}
    </style>
    <script>
        function toggleAccordion(header) {{
            const body = header.nextElementSibling;
            const toggle = header.querySelector('.accordion-toggle');
            if (body.style.display === 'none') {{
                body.style.display = 'block';
                toggle.style.transform = 'rotate(180deg)';
            }} else {{
                body.style.display = 'none';
                toggle.style.transform = 'rotate(0deg)';
            }}
        }}

        function toggleActivity(card) {{
            const body = card.querySelector('.activity-body');
            const toggle = card.querySelector('.activity-toggle');
            if (body.style.display === 'none') {{
                body.style.display = 'block';
                toggle.style.transform = 'rotate(180deg)';
            }} else {{
                body.style.display = 'none';
                toggle.style.transform = 'rotate(0deg)';
            }}
        }}

        function switchTab(button, tabId) {{
            const container = button.closest('.tabs-container');
            container.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            container.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
            button.classList.add('active');
            document.getElementById(tabId).style.display = 'block';
        }}

        function prevSlide(button) {{
            const container = button.closest('.carousel-container');
            const slides = container.querySelectorAll('.carousel-slide');
            let current = -1;
            slides.forEach((slide, idx) => {{
                if (slide.style.display === 'block') current = idx;
                slide.style.display = 'none';
            }});
            const prev = current === 0 ? slides.length - 1 : current - 1;
            slides[prev].style.display = 'block';
        }}

        function nextSlide(button) {{
            const container = button.closest('.carousel-container');
            const slides = container.querySelectorAll('.carousel-slide');
            let current = -1;
            slides.forEach((slide, idx) => {{
                if (slide.style.display === 'block') current = idx;
                slide.style.display = 'none';
            }});
            const next = (current + 1) % slides.length;
            slides[next].style.display = 'block';
        }}
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>{escape_html(course_state.title)}</h1>
            <p class="course-meta">{escape_html(course_state.description)}</p>
            <p class="course-meta">📚 {len(course_state.modules)} Módulos •
               {sum(len(sm.sections) for m in course_state.modules for sm in m.submodules)} Secciones</p>
        </header>
"""

    # Generate Table of Contents
    html += '<div class="table-of-contents">'
    html += "<h2>📑 Índice del Curso</h2>"

    section_counter = 0
    for module_idx, module in enumerate(course_state.modules, 1):
        html += '<div class="toc-module">'
        html += f'<div class="toc-module-title">Módulo {module_idx}: {escape_html(module.title)}</div>'

        for submodule_idx, submodule in enumerate(module.submodules, 1):
            html += '<div class="toc-submodule">'
            html += (
                f'<div class="toc-submodule-title">{module_idx}.{submodule_idx} {escape_html(submodule.title)}</div>'
            )
            html += '<ul class="toc-sections">'

            for section_idx, section in enumerate(submodule.sections, 1):
                section_counter += 1
                html += f'<li><a href="#section-{section_counter}">{module_idx}.{submodule_idx}.{section_idx} {escape_html(section.title)}</a></li>'

            html += "</ul>"
            html += "</div>"

        html += "</div>"

    html += "</div>"

    # Render all modules
    section_counter = 0
    for module_idx, module in enumerate(course_state.modules, 1):
        html += '<div class="module">'
        html += f'<h2 class="module-title">Módulo {module_idx}: {escape_html(module.title)}</h2>'

        for submodule_idx, submodule in enumerate(module.submodules, 1):
            html += '<div class="submodule">'
            html += f'<h3 class="submodule-title">{module_idx}.{submodule_idx} {escape_html(submodule.title)}</h3>'
            if submodule.description and submodule.description != submodule.title:
                html += f'<p class="submodule-description">{escape_html(submodule.description)}</p>'

            for section in submodule.sections:
                section_counter += 1
                html += render_section(section, section_counter)

            html += "</div>"

        html += "</div>"

    html += """
        <footer>
            <p>📚 Curso generado por Course Generator Agent</p>
        </footer>
    </div>
</body>
</html>
"""

    # Fix local image paths to be relative to the HTML file
    import os
    import re as _re

    html_dir = os.path.dirname(os.path.abspath(output_path))

    def _fix_src(match):
        prefix, src, quote = match.group(1), match.group(2), match.group(3)
        if not src.startswith(("http://", "https://")):
            src = _make_image_path_relative(src, html_dir)
        return f"{prefix}{src}{quote}"

    html = _re.sub(r'(<img[^>]+src=")([^"]+)(")', _fix_src, html)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
