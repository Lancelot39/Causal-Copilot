from html import escape
from typing import Mapping, Sequence


APP_JS = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.display = 'flex';
    container.style.justifyContent = 'center';
    container.style.alignItems = 'center';
    container.style.marginBottom = '20px';
    container.style.position = 'relative';

    // Title container
    var titleContainer = document.createElement('div');
    titleContainer.style.fontSize = '2em';
    titleContainer.style.fontWeight = 'bold';
    titleContainer.style.textAlign = 'center';
    titleContainer.style.display = 'flex';
    titleContainer.style.alignItems = 'center';
    titleContainer.style.justifyContent = 'center';

    var text = 'Welcome to Causal Copilot!';
    var animatedText = document.createElement('div');
    animatedText.style.display = 'inline-block';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.1s';
                letter.innerText = text[i];
                animatedText.appendChild(letter);
                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 30);
            }, i * 100);
        })(i);
    }
    titleContainer.appendChild(animatedText);
    // Video button - create but don't append until animation completes
    var videoBtn = document.createElement('button');
    videoBtn.setAttribute('aria-label', 'Watch walk-through video on YouTube');
    videoBtn.style.marginLeft = '12px';
    videoBtn.style.height = '24px';
    videoBtn.style.padding = '0 8px';
    videoBtn.style.borderRadius = '4px';
    videoBtn.style.border = 'none';
    videoBtn.style.background = '#1976d2';
    videoBtn.style.display = 'flex';
    videoBtn.style.alignItems = 'center';
    videoBtn.style.justifyContent = 'center';
    videoBtn.style.cursor = 'pointer';
    videoBtn.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
    videoBtn.style.transition = 'background 0.2s, box-shadow 0.2s';
    videoBtn.style.color = '#fff';
    videoBtn.style.fontWeight = 'normal';
    videoBtn.style.fontSize = '0.35em';
    videoBtn.style.letterSpacing = '0.01em';
    videoBtn.onmouseover = function() {
        videoBtn.style.background = '#0d5ca1';
        videoBtn.style.boxShadow = '0 1px 4px rgba(0,0,0,0.15)';
    };
    videoBtn.onmouseout = function() {
        videoBtn.style.background = '#1976d2';
        videoBtn.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
    };
    videoBtn.onclick = function() {
        window.open('https://www.youtube.com/watch?si=3DTT2AlEIcAf-T_E&v=U9-b0ZqqM24&feature=youtu.be', '_blank');
    };
    videoBtn.innerText = '📺 walk-through video';
    videoBtn.style.opacity = '0';
    videoBtn.style.transition = 'opacity 0.5s';

    // Wait for the text animation to complete before showing the button
    setTimeout(function() {
        titleContainer.appendChild(videoBtn);
        setTimeout(function() {
            videoBtn.style.opacity = '1';
        }, 50);
    }, text.length * 100 + 100); // Wait for all letters plus a small buffer

    container.appendChild(titleContainer);

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);
    return 'Animation created';
}
"""

APP_CSS = """
.cc-app-shell {
    max-width: 1440px;
    margin: 0 auto;
}
.cc-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.cc-panel {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #ffffff;
}
.input-buttons {
    position: absolute !important;
    right: 10px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    display: flex !important;
    gap: 5px !important;
}
.icon-button {
    padding: 0 !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 16px !important;
    background: transparent !important;
}
.icon-button:hover {
    background: #f0f0f0 !important;
}
.icon {
    width: 20px;
    height: 20px;
    margin: 6px;
    display: inline-block;
    vertical-align: middle;
}
.message-wrap {
    display: flex !important;
    align-items: flex-start !important;
    gap: 10px !important;
    padding: 15px !important;
}
.avatar {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 20px !important;
}
.bot-avatar {
    background: #e3f2fd !important;
    color: #1976d2 !important;
}
.user-avatar {
    background: #f5f5f5 !important;
    color: #333 !important;
}
.message {
    padding: 12px 16px !important;
    border-radius: 12px !important;
    max-width: 100% !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    object-fit: contain !important;
}
.bot-message {
    background: #e3f2fd !important;
    margin-right: auto !important;
}
.user-message {
    background: #f5f5f5 !important;
    margin-left: auto !important;
}
/* Gallery Section Styles */
.gallery-section {
    margin-top: 40px;
    margin-bottom: 20px;
}
.gallery-heading {
    width: 100%;
    text-align: center;
    font-size: 28px;
    margin-bottom: 20px;
}
.filter-buttons {
    display: flex;
    justify-content: center;
    margin-bottom: 30px;
    flex-wrap: wrap;
    gap: 10px;
}
.filter-btn {
    border-radius: 20px;
    background-color: #f5f5f5;
    padding: 8px 16px;
    transition: all 0.3s ease;
}
.filter-btn.active {
    background-color: #333;
    color: white;
}
.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 20px;
    margin: 0 auto;
    padding: 0 20px;
}
.gallery-card {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
}
.gallery-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
}
.card-img-container {
    width: 100%;
    height: 200px;
    overflow: hidden;
}
.gallery-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.card-content {
    padding: 15px;
}
.card-title {
    font-size: 18px;
    font-weight: bold;
    margin: 0;
    margin-bottom: 10px;
}
.card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card-author {
    font-size: 14px;
    color: #555;
}
.card-likes {
    margin-left: auto;
    font-size: 14px;
    color: #ff4757;
}
/* Responsive Adjustments */
@media (max-width: 768px) {
    .gallery-container {
        flex-direction: column;
    }
    .gallery-card {
        width: 100%;
        margin-bottom: 20px;
    }
}
.report-gallery {
    display: grid;
    grid-template-columns: repeat(6, 1fr); /* Changed to 6 columns for a single row */
    gap: 15px; /* Reduced gap to fit better */
    margin: 20px auto;
    max-width: 1400px; /* Increased width to accommodate all 6 cards */
}
.report-card-link {
    text-decoration: none;
    color: inherit;
}
.report-card {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    position: relative;
    cursor: pointer;
}
.report-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 20px rgba(0,0,0,0.15);
}
/* PDF Icon */
.pdf-icon {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 32px;
    height: 32px;
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    z-index: 3;
}
.pdf-icon svg {
    width: 18px;
    height: 18px;
    fill: #e74c3c;
}
/* Card Image Area (Always Visible) */
.report-card-image-area {
    width: 100%;
    height: 140px; /* Reduced height for better fit in single row */
    background-color: #f0f0f0;
    background-size: cover;
    background-position: center;
    overflow: hidden;
}
.report-card-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.report-card-content {
    padding: 15px; /* Reduced padding */
}
.report-card-title {
    font-size: 16px; /* Smaller font */
    font-weight: bold;
    margin: 0 0 8px 0;
    color: #333;
}
.report-card-desc {
    font-size: 12px; /* Smaller font */
    color: #666;
    margin-bottom: 10px;
    line-height: 1.3;
}
.report-card-author {
    font-size: 12px; /* Smaller font */
    color: #555;
    font-style: italic;
}
/* Removed hover thumbnail styles */
.report-card::before {
   /* Optional: Keep the top accent bar */
   content: '';
   position: absolute;
   top: 0;
   left: 0;
   width: 100%;
   height: 5px;
   background: linear-gradient(90deg, #3498db, #2980b9);
   z-index: 2;
}
.message-bubble img {
    pointer-events: none !important;
}

/* Optional: Also adjust image sizing */
.message-bubble img {
    max-width: none !important;
    max-height: none !important;
    width: 400px !important;
}

/* Hide the Gradio footer */
footer{display:none !important}
"""


def build_welcome_message() -> str:
    return (
        "Welcome to Causal Copilot. Upload a CSV dataset or choose a demo "
        "dataset to begin causal discovery."
    )


_PDF_ICON_HTML = """
                    <div class="pdf-icon" aria-hidden="true">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" focusable="false">
                            <path d="M181.9 256.1c-5-16-4.9-46.9-2-46.9 8.4 0 7.6 36.9 2 46.9zm-1.7 47.2c-7.7 20.2-17.3 43.3-28.4 62.7 18.3-7 39-17.2 62.9-21.9-12.7-9.6-24.9-23.4-34.5-40.8zM86.1 428.1c0 .8 13.2-5.4 34.9-40.2-6.7 6.3-16.8 15.8-24.1 26.3-10.7 15.5-11.6 14-10.8 13.9zm28.6-181.9c7.4-6.5 21.6-10.7 38.8-13.3 4.9-14.7 8.4-33.4 8.4-33.4s-13.6 8.1-29.8 10.2c-14.1 1.9-24.5 7.3-34.1 20.1-9.8 13.1-8.1 10.4-8.1 10.4s9.4-6.2 24.8-6z"/>
                            <path d="M384 121.9v6.1H256V0h6.1c6.4 0 12.5 2.5 17 7l97.9 98c4.5 4.5 7 10.6 7 16.9zM248 160h136v328c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V24C0 10.7 10.7 0 24 0h200v136c0 13.2 10.8 24 24 24zm-84.5 98.6c19.3-5.7 45.5-10.4 67.7-13.2-7.9-35.6-10.3-53.3-10.3-53.3s-16 2.8-37.9 7.7c-36.3 7.9-38.9 9-55.8 31.1-6 7.9-9.5 14.4-12.7 19.5-18.7 26.6-41.7 51.7-52.6 66.1-5.5 7.3-13.3 18.3-19.6 22.2-9.1 5.7-21.1 1.7-23.7-7.8-2.6-9.5 4.3-19.5 9.5-22.7 3.5-2.1 8.7-9.1 15.8-20.3 10.8-15.7 19.1-33.3 19.1-33.3s-10.6 6.7-13.4 10.5c-14.9 19.7-40.8 49-51.9 73.3-10.8 24.4-4.9 37.3 8.1 42.8 9.2 3.8 21.3-4 29.5-12.6 11.7-12.1 17.9-20.2 28.3-38.8 18.1-32.3 30.6-62.8 30.6-62.8s-.3 14.2-.8 22.7c-.7 12.3-2.8 14.8-7.9 20.3-5.8 6.2-13.5 6.6-19.9 6.4-8.5-.4-10.4 5.8-7.4 9.1 9 10.2 29.2 6.3 41.4-9.2 13.1-16.6 11.6-37.8 11-46.9-1.4-21.6 2.4-49.5 2.4-49.5z"/>
                        </svg>
                    </div>
"""


def build_report_gallery_html(report_items: Sequence[Mapping[str, str]]) -> str:
    cards = []
    for item in report_items:
        title = escape(item["title"])
        description = escape(item["description"])
        author = escape(item["author"])
        file_url = escape(item["file"], quote=True)
        image_url = escape(item["image"], quote=True)
        cards.append(
            f"""
            <a class="report-card-link" href="{file_url}" target="_blank" rel="noopener noreferrer">
                <article class="report-card" data-file="{file_url}">
{_PDF_ICON_HTML}
                    <div class="report-card-image-area">
                        <img class="report-card-image" src="{image_url}" alt="{title} report cover">
                    </div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">{title}</h3>
                        <p class="report-card-desc">{description}</p>
                        <p class="report-card-author">By {author}</p>
                    </div>
                </article>
            </a>
            """
        )
    return '<div class="report-gallery">' + "\n".join(cards) + "</div>"
