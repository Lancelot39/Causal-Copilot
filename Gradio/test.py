import gradio as gr
import os

# Get the absolute path to the public directory
current_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(current_dir, "public")

# Create your Gradio interface
with gr.Blocks(css="style.css") as demo:
    # Gallery section heading
    with gr.Row(elem_classes=["gallery-section"]):
        gr.Markdown("## 探索不同领域的因果分析案例研究。", elem_classes=["gallery-heading"])
    
    # Define report cards with real PDF reports
    with gr.Row():
        # Use direct HTML instead of generating it dynamically
        # This ensures we have exactly the right format
        gr.HTML("""
        <style>
        .report-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 25px;
            margin: 20px auto;
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
        .report-card-image-area {
            width: 100%;
            height: 180px;
            background-color: #f0f0f0;
            background-size: cover;
            background-position: center;
        }
        .report-card-content {
            padding: 20px;
        }
        .report-card-title {
            font-size: 20px;
            font-weight: bold;
            margin: 0 0 10px 0;
            color: #333;
        }
        .report-card-desc {
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            line-height: 1.4;
        }
        .report-card-author {
            font-size: 14px;
            color: #555;
            font-style: italic;
        }
        .report-card::before {
           content: '';
           position: absolute;
           top: 0;
           left: 0;
           width: 100%;
           height: 5px;
           background: linear-gradient(90deg, #3498db, #2980b9);
           z-index: 2; 
        }
        </style>
        
        <div class="report-gallery">
            <!-- Abalone Report Card -->
            <a href="/file=abalone.jpg" target="_blank" style="text-decoration: none; color: inherit;"> 
                <div class="report-card">
                    <div class="report-card-image-area" style="background-image: url('/file=abalone.jpg');"></div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">Abalone Causal Analysis</h3>
                        <p class="report-card-desc">Discovering relationships between physical attributes and age of abalone</p>
                        <p class="report-card-author">By Marine Biology Team</p>
                    </div>
                </div>
            </a>
            
            <!-- Heart Disease Report Card -->
            <a href="/file=tabular-heartdisease.pdf" target="_blank" style="text-decoration: none; color: inherit;"> 
                <div class="report-card">
                    <div class="report-card-image-area" style="background-image: url('/file=tabular-heartdisease.pdf');"></div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">Heart Disease Study</h3>
                        <p class="report-card-desc">Causal factors influencing heart disease development</p>
                        <p class="report-card-author">By Medical Research Group</p>
                    </div>
                </div>
            </a>
            
            <!-- Climate Time Series Card -->
            <a href="/file=timeseries-climate.pdf" target="_blank" style="text-decoration: none; color: inherit;"> 
                <div class="report-card">
                    <div class="report-card-image-area" style="background-image: url('/file=timeseries-climate.pdf');"></div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">Climate Time Series Analysis</h3>
                        <p class="report-card-desc">Temporal patterns and causality in climate data</p>
                        <p class="report-card-author">By Environmental Sciences Lab</p>
                    </div>
                </div>
            </a>
            
            <!-- Student Performance Card -->
            <a href="/file=tabular-student-score.pdf" target="_blank" style="text-decoration: none; color: inherit;"> 
                <div class="report-card">
                    <div class="report-card-image-area" style="background-image: url('/file=tabular-student-score.pdf');"></div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">Student Performance Factors</h3>
                        <p class="report-card-desc">Determining key influences on academic achievement</p>
                        <p class="report-card-author">By Education Research Center</p>
                    </div>
                </div>
            </a>
            
            <!-- Earthquake Time Series Card -->
            <a href="/file=timeseries-earthquake.pdf" target="_blank" style="text-decoration: none; color: inherit;"> 
                <div class="report-card">
                    <div class="report-card-image-area" style="background-image: url('/file=timeseries-earthquake.pdf');"></div>
                    <div class="report-card-content">
                        <h3 class="report-card-title">Earthquake Time Series</h3>
                        <p class="report-card-desc">Temporal factors affecting seismic activity</p>
                        <p class="report-card-author">By Geological Survey Team</p>
                    </div>
                </div>
            </a>
        </div>
        """)

# Launch with explicit allowed_paths using absolute path
demo.launch(
    allowed_paths=[public_dir],  # Use absolute path
    server_name="0.0.0.0"
)