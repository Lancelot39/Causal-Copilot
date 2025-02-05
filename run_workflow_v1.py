# import time
# import gradio as gr
# import pandas as pd
# import io
# import os
# import shutil
# from datetime import datetime
# import sys
# from queue import Queue

# from pydantic import BaseModel

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Gradio.demo_config import get_demo_config
# from global_setting.Initialize_state import global_state_initialization
# from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
# from preprocess.dataset import knowledge_info
# from preprocess.eda_generation import EDA
# from algorithm.filter import Filter
# from algorithm.program import Programming
# from algorithm.rerank import Reranker
# from postprocess.judge import Judge
# from postprocess.visualization import Visualization
# from postprocess.report_generation import Report_generation

# Global variables
# UPLOAD_FOLDER = "./demo_data"

chat_history = []
target_path = None
output_dir = None
REQUIRED_INFO = {
    'data_uploaded': False,
    'initial_query': False,
}
MAX_CONCURRENT_REQUESTS = 5
MAX_QUEUE_SIZE = 10

# Demo dataset configs
DEMO_DATASETS = {
    "Abalone": {
        "name": "🐚 Real Dataset:Abalone",
        "path": "dataset/Abalone/Abalone.csv",
        "query": "YES. Find causal relationships between physical measurements and age of abalone. The dataset contains numerical measurements of physical characteristics.",
    },
    "Sachs": {
        "name": "🧬 Real Dataset: Sachs",
        "path": "dataset/sachs/sachs.csv", 
        "query": "YES. Discover causal relationships between protein signaling molecules. The data contains flow cytometry measurements of proteins and phospholipids."
    },
    "CCS Data": {
        "name": "📊 Real Dataset: CCS Data",
        "path": "dataset/CCS_Data/CCS_Data.csv",
        "query": "YES. Analyze causal relationships between variables in the CCS dataset. The data contains multiple continuous variables."
    },
    "Ozone": {
        "name": "🌫️ Real Dataset: Ozone", 
        "path": "dataset/Ozone/Ozone.csv",
        "query": "YES. This is a Time-Series dataset, investigate causal factors affecting ozone levels. The data contains atmospheric and weather measurements over time."
    },
    "Linear_Gaussian": {
        "name": "🟦 Simualted Data: Linear Gaussian",
        "path": "dataset/Linear_Gaussian/Linear_Gaussian_data.csv",
        "query": "NO. The data follows linear relationships with Gaussian noise. Please discover the causal structure."
    },
    "Linear_Nongaussian": {
        "name": "🟩 Simulated Data: Linear Non-Gaussian",
        "path": "dataset/Linear_Nongaussian/Linear_Nongaussian_data.csv", 
        "query": "NO. The data follows linear relationships with non-Gaussian noise. Please discover the causal structure."
    }
}


# def upload_file(file):
#     # TODO: add more complicated file unique ID handling
#     global target_path, output_dir

#     date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', '')), exist_ok=True)

#     target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''),
#                                os.path.basename(file.name))
#     output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''))
#     shutil.copy(file.name, target_path)
#     return target_path


# def handle_file_upload(file, chatbot, file_upload_btn, download_btn):
#     chatbot = chatbot.copy()
#     try:
#         global REQUIRED_INFO
#         if file.name.endswith('.csv'):
#             df = pd.read_csv(file.name)
#             upload_file(file)
#             REQUIRED_INFO['data_uploaded'] = True
#             bot_message = (f"✅ Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns! \n"
#                            "🤔 Please follow the guidances above for your initial query. \n"
#                            "✨ It would be super helpful if you can include more relevant information, e.g., background/context/prior/statistical information!")
#         else:
#             bot_message = "❌ Please upload a CSV file."
#         chatbot.append((None, bot_message))
#         return chatbot, file_upload_btn, download_btn

#     except Exception as e:
#         error_message = f"❌ Error loading file: {str(e)}"
#         chatbot.append((None, error_message))
#         return chatbot, file_upload_btn, download_btn


# def process_initial_query(message):
#     global REQUIRED_INFO
#     # TODO: check if the initial query is valid or satisfies the requirements
#     REQUIRED_INFO['initial_query'] = True
#     if not REQUIRED_INFO['initial_query']:
#         chat_history.append((None,
#                              "Please enter your initial query first before proceeding. It would be helpful to provide some information about the background/context/prior/statistical information about the dataset."))


# def process_message(message, chat_history, download_btn):
#     global target_path, REQUIRED_INFO

#     REQUIRED_INFO['processing'] = True
#     process_initial_query(message)

#     if not REQUIRED_INFO['data_uploaded']:
#         print('not uploaded')
#         chat_history.append((message, "Please upload your dataset first before proceeding."))
#         return chat_history, download_btn

#     if not REQUIRED_INFO['initial_query']:
#         print('not query')
#         chat_history.append((message, "Please input your initial query."))
#         return chat_history, download_btn

#     try:
#         # Initialize config and global state
#         config = get_demo_config()
#         config.data_file = target_path
#         config.initial_query = message

#         args = type('Args', (), {})()
#         for key, value in config.__dict__.items():
#             setattr(args, key, value)

#         if 'YES' in message:
#             args.data_mode = 'real'
#         elif 'NO' in message:
#             args.data_mode = 'simulated'
#         else:
#             print('not feature indicator')
#             chat_history.append((message,
#                                  "Please indicate if your dataset has meaningful feature names using 'YES' or 'NO', which would help us generate appropriate report for you."))
#             yield chat_history, download_btn

#         # Add user message
#         # chat_history.append((message, None))
#         # chat_history.append(("🔄 Initializing analysis pipeline...", None))
#         global_state = global_state_initialization(args)

#         # Load data
#         # chat_history.append((None, "📊 Loading and preprocessing data..."))
#         global_state.user_data.raw_data = pd.read_csv(target_path)
#         global_state.user_data.processed_data = global_state.user_data.raw_data
#         # chat_history.append((None, "✅ Data loaded successfully"))
#         yield chat_history, download_btn

#         chat_history.append(
#             (f"📈 Run statistical analysis on Dataset {target_path.split('/')[-1].replace('.csv', '')}...", None))
#         yield chat_history, download_btn

#         user_linear = global_state.statistics.linearity
#         user_gaussian = global_state.statistics.gaussian_error

#         global_state = stat_info_collection(global_state)
#         global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

#         if global_state.statistics.data_type == "Continuous":
#             if user_linear is None:
#                 chat_history.append(("✍️ Generate residuals plots ...", None))
#                 yield chat_history, download_btn
#                 chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/residuals_plot.jpg',)))
#                 yield chat_history, download_btn
#             if user_gaussian is None:
#                 chat_history.append(("✍️ Generate Q-Q plots ...", None))
#                 yield chat_history, download_btn
#                 chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/qq_plot.jpg',)))
#                 yield chat_history, download_btn

#         chat_history.append((None, global_state.statistics.description))
#         yield chat_history, download_btn

#         # Knowledge generation
#         if args.data_mode == 'real':
#             chat_history.append(("🌍 Generate background knowledge based on the dataset you provided...", None))
#             yield chat_history, download_btn
#             global_state = knowledge_info(args, global_state)

#             knowledge_clean = str(global_state.user_data.knowledge_docs).replace("[", "").replace("]", "").replace('"',
#                                                                                                                    "").replace(
#                 "\\n\\n", "\n\n").replace("\\n", "\n")
#             chat_history.append((None, knowledge_clean))
#             yield chat_history, download_btn
#         elif args.data_mode == 'simulated':
#             global_state = knowledge_info(args, global_state)

#         # EDA Generation
#         chat_history.append(("🔍 Run exploratory data analysis...", None))
#         yield chat_history, download_btn
#         my_eda = EDA(global_state)
#         my_eda.generate_eda()
#         chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_corr.jpg',)))
#         chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_dist.jpg',)))
#         yield chat_history, download_btn

#         # Algorithm Selection
#         if global_state.algorithm.selected_algorithm is None:
#             chat_history.append(("🤖 Select optimal causal discovery algorithm and its hyperparameter...", None))
#             yield chat_history, download_btn
#             filter = Filter(args)
#             global_state = filter.forward(global_state)
#             reranker = Reranker(args)
#             global_state = reranker.forward(global_state)
#             chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
#             chat_history.append((None, f"🤔 Algorithm selection reasoning: {global_state.algorithm.selected_reason}"))
#         else:
#             chat_history.append(
#                 ("🤖 Select optimal hyperparameter for your selected causal discovery algorithm...", None))
#             yield chat_history, download_btn

#             filter = Filter(args)
#             global_state = filter.forward(global_state)
#             reranker = Reranker(args)
#             global_state = reranker.forward(global_state)
#             chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))

#         hyperparameter_text = ""
#         for param, details in global_state.algorithm.algorithm_arguments_json['hyperparameters'].items():
#             value = details['value']
#             explanation = details['explanation']
#             hyperparameter_text += f"  Parameter: {param}\n"
#             hyperparameter_text += f"  Value: {value}\n"
#             hyperparameter_text += f"  Explanation: {explanation}\n\n"
#         chat_history.append(
#             (None,
#              f"📖 Hyperparameters for the selected algorithm {global_state.algorithm.selected_algorithm}: \n\n {hyperparameter_text}"))
#         yield chat_history, download_btn

#         # Causal Discovery
#         chat_history.append(("🔄 Run causal discovery algorithm...", None))
#         yield chat_history, download_btn
#         programmer = Programming(args)
#         global_state = programmer.forward(global_state)
#         # Visualization for Initial Graph
#         chat_history.append(("📊 Generate causal graph visualization...", None))
#         yield chat_history, download_btn
#         my_visual_initial = Visualization(global_state)
#         pos = my_visual_initial.get_pos(global_state.results.raw_result)
#         if global_state.user_data.ground_truth is not None:
#             my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.jpg', pos)
#             my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', pos)
#             chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/true_graph.jpg',)))
#             yield chat_history, download_btn
#         if global_state.results.raw_result is not None:
#             my_visual_initial.plot_pdag(global_state.results.raw_result, 'initial_graph.jpg', pos)
#             my_visual_initial.plot_pdag(global_state.results.raw_result, 'initial_graph.pdf', pos)
#             chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/initial_graph.jpg',)))
#             yield chat_history, download_btn
#             my_report = Report_generation(global_state, args)
#             global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
#             print('graph analysis', global_state.logging.graph_conversion['initial_graph_analysis'])
#             chat_history.append((None, global_state.logging.graph_conversion['initial_graph_analysis']))
#             yield chat_history, download_btn
#         # Evaluation for Initial Graph
        
#         chat_history.append(("📝 Evaluate and Revise the initial result...", None))
#         yield chat_history, download_btn
#         try:
#             judge = Judge(global_state, args)
#             global_state = judge.forward(global_state)
#         except Exception as e:
#             print('error during judging:', e)
#             judge = Judge(global_state, args)
#             global_state = judge.forward(global_state)
#         my_visual_revise = Visualization(global_state)
#         if args.data_mode=='real':
#             # Plot Revised Graph
#             if global_state.results.revised_graph is not None:
#                 my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.pdf', pos)
#                 my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.jpg', pos)
#                 chat_history.append((None, f"This is the revised graph with Bootstrap and LLM techniques"))
#                 yield chat_history, download_btn
#                 chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/revised_graph.jpg',)))
#                 yield chat_history, download_btn
#         # Plot Bootstrap Heatmap
#         paths = my_visual_revise.boot_heatmap_plot()
#         chat_history.append(
#             (None, f"The following heatmaps show the confidence probability we have on different kinds of edges"))
#         yield chat_history, download_btn
#         for path in paths:
#             chat_history.append((None, (path,)))
#             yield chat_history, download_btn

#         chat_history.append((None, "✅ Causal discovery analysis completed"))
#         yield chat_history, download_btn

#         # Report Generation
#         chat_history.append(("📝 Generate comprehensive report and it may take a few minutes, stay tuned...", None))
#         yield chat_history, download_btn
#         report_gen = Report_generation(global_state, args)
#         report = report_gen.generation(debug=False)
#         report_gen.save_report(report)
#         report_path = os.path.join(output_dir, 'output_report', 'report.pdf')
#         while not os.path.isfile(report_path):
#             chat_history.append((None,
#                                  "❌ An error occurred during the Report Generation, we are trying again and please wait for a few minutes."))
#             yield chat_history, download_btn
#             report_gen = Report_generation(global_state, args)
#             report = report_gen.generation(debug=False)
#             report_gen.save_report(report)

#         # Final steps
#         chat_history.append((None, "🎉 Analysis complete!"))
#         chat_history.append((None, "📥 You can now download your detailed report using the download button below."))

#         REQUIRED_INFO['processing'] = False

#         download_btn = gr.DownloadButton(
#             "📥 Download Exclusive Report",
#             size="sm",
#             elem_classes=["icon-button"],
#             scale=1,
#             value=os.path.join(output_dir, 'output_report', 'report.pdf'),
#             interactive=True
#         )
#         yield chat_history, download_btn

#         chat_history.append((None, ""))
#         return chat_history, download_btn

#     except Exception as e:
#         chat_history.append((None, f"❌ An error occurred during analysis: {str(e)}, please try again"))
#         print(str(e))
#         import traceback
#         traceback.print_exc()
#         yield chat_history, download_btn
#         return chat_history, download_btn


# def clear_chat():
#     global target_path, REQUIRED_INFO, output_dir, chat_history
#     # Reset global variables
#     target_path = None
#     output_dir = None
#     chat_history = []

#     # Reset required info flags
#     REQUIRED_INFO['data_uploaded'] = False
#     REQUIRED_INFO['initial_query'] = False

#     # Return initial welcome message
#     return [(None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today? \n"
#                    "⏫ Some guidances before uploading your dataset: \n"
#                    "1️⃣ The dataset should be tabular in .csv format, with each column representing a variable. \n "
#                    "2️⃣ Ensure that the features are in numerical format or appropriately encoded if categorical. \n"
#                    "3️⃣ For initial query, your dataset has meaningful feature names, please indicate it using 'YES' or 'NO'. \n"
#                    "4️⃣ Please mention heterogeneity and its indicator's column name in your initial query if there is any. \n"
#                    "💡 Example initial query: 'YES. Use PC algorithm to analyze causal relationships between variables. The dataset has heterogeneity with domain column named 'country'.' \n")],


# def load_demo_dataset(dataset_name, chatbot, demo_btn, download_btn):
#     global target_path, REQUIRED_INFO, output_dir
#     dataset = DEMO_DATASETS[dataset_name]
#     source_path = dataset["path"]

#     date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', '')),
#                 exist_ok=True)

#     target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''),
#                                os.path.basename(source_path))
#     output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''))
#     shutil.copy(source_path, target_path)

#     REQUIRED_INFO['data_uploaded'] = True
#     REQUIRED_INFO['initial_query'] = True

#     df = pd.read_csv(target_path)
#     chatbot.append((f"{dataset['query']}", None))
#     bot_message = f"✅ Loaded demo dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns."
#     chatbot = chatbot.copy()
#     chatbot.append((None, bot_message))
#     return chatbot, demo_btn, download_btn, dataset['query']

# def init0():
#     js = """
#     function createGradioAnimation() {
#         var container = document.createElement('div');
#         container.id = 'gradio-animation';
#         container.style.fontSize = '2em';
#         container.style.fontWeight = 'bold';
#         container.style.textAlign = 'center';
#         container.style.marginBottom = '20px';

#         var text = 'Welcome to Causal Copilot!';
#         for (var i = 0; i < text.length; i++) {
#             (function(i){
#                 setTimeout(function(){
#                     var letter = document.createElement('span');
#                     letter.style.opacity = '0';
#                     letter.style.transition = 'opacity 0.5s';
#                     letter.innerText = text[i];

#                     container.appendChild(letter);

#                     setTimeout(function() {
#                         letter.style.opacity = '1';
#                     }, 50);
#                 }, i * 250);
#             })(i);
#         }

#         var gradioContainer = document.querySelector('.gradio-container');
#         gradioContainer.insertBefore(container, gradioContainer.firstChild);

#         return 'Animation created';
#     }
#     """

#     with gr.Blocks(js=js, theme=gr.themes.Soft(), css="""
#         .input-buttons { 
#             position: absolute !important; 
#             right: 10px !important;
#             top: 50% !important;
#             transform: translateY(-50%) !important;
#             display: flex !important;
#             gap: 5px !important;
#         }
#         .icon-button { 
#             padding: 0 !important;
#             width: 32px !important;
#             height: 32px !important;
#             border-radius: 16px !important;
#             background: transparent !important;
#         }
#         .icon-button:hover { 
#             background: #f0f0f0 !important;
#         }
#         .icon {
#             width: 20px;
#             height: 20px;
#             margin: 6px;
#             display: inline-block;
#             vertical-align: middle;
#         }
#         .message-wrap {
#             display: flex !important;
#             align-items: flex-start !important;
#             gap: 10px !important;
#             padding: 15px !important;
#         }
#         .avatar {
#             width: 40px !important;
#             height: 40px !important;
#             border-radius: 50% !important;
#             display: flex !important;
#             align-items: center !important;
#             justify-content: center !important;
#             font-size: 20px !important;
#         }
#         .bot-avatar {
#             background: #e3f2fd !important;
#             color: #1976d2 !important;
#         }
#         .user-avatar {
#             background: #f5f5f5 !important;
#             color: #333 !important;
#         }
#         .message {
#             padding: 12px 16px !important;
#             border-radius: 12px !important;
#             max-width: 100% !important;
#             box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
#             object-fit: contain !important;
#         }
#         .bot-message {
#             background: #e3f2fd !important;
#             margin-right: auto !important;
#         }
#         .user-message {
#             background: #f5f5f5 !important;
#             margin-left: auto !important;
#         }
#     """) as demo:
#         chatbot = gr.Chatbot(
#             value=[
#                 (None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today? \n"
#                     "⏫ Some guidances before uploading your dataset: \n"
#                     "1️⃣ The dataset should be tabular in .csv format, with each column representing a variable. \n "
#                     "2️⃣ Ensure that the features are in numerical format or appropriately encoded if categorical. \n"
#                     "3️⃣ For initial query, your dataset has meaningful feature names, please indicate it using 'YES' or 'NO'. \n"
#                     "4️⃣ Please mention heterogeneity and its indicator's column name in your initial query if there is any. \n"
#                     "💡 Example initial query: 'YES. Use PC algorithm to analyze causal relationships between variables. The dataset has heterogeneity with domain column named 'country'.' \n")],
#             height=700,
#             show_label=False,
#             show_share_button=False,
#             avatar_images=["https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f600.png",
#                         "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f916.png"],
#             bubble_full_width=False,
#             elem_classes=["message-wrap"],
#             render_markdown=True
#         )


#         def disable_all_inputs(dataset_name, chatbot, clicked_btn, download_btn, msg, all_demo_buttons):
#             """Disable all interactive elements"""
#             updates = []
#             for _ in range(int(all_demo_buttons)):
#                 updates.append(gr.update(interactive=False))
#             updates.extend([
#                 gr.update(interactive=False),  # For download button
#                 gr.update(value="", interactive=False),  # For textbox
#                 gr.update(interactive=False),  # For file upload
#                 gr.update(interactive=False),  # For reset button
#             ])
#             return updates


#         def enable_all_inputs(all_demo_buttons):
#             """Re-enable all interactive elements"""
#             updates = [
#                 gr.update(interactive=True) for _ in range(int(all_demo_buttons))  # For all demo buttons
#             ]
#             updates.extend([
#                 gr.update(interactive=True),  # For download button
#                 gr.update(value="", interactive=True),  # For textbox
#                 gr.update(interactive=True),  # For file upload
#                 gr.update(interactive=True),  # For reset button
#             ])
#             return updates


#         with gr.Row():
#             with gr.Column(scale=24):
#                 with gr.Row():
#                     msg = gr.Textbox(
#                         placeholder="Enter text here",
#                         elem_classes="input-box",
#                         show_label=False,
#                         container=False,
#                         scale=12
#                     )
#                     file_upload = gr.UploadButton(
#                         "📎 Upload Your Data (.csv)",
#                         file_types=[".csv"],
#                         size="sm",
#                         elem_classes=["icon-button"],
#                         scale=5,
#                         file_count="single"
#                     )
#                     download_btn = gr.DownloadButton(
#                         "📥 Download Exclusive Report",
#                         size="sm",
#                         elem_classes=["icon-button"],
#                         scale=6,
#                         interactive=False
#                     )
#                     reset_btn = gr.Button("🔄 Reset", scale=1, elem_classes=["icon-button"], size="sm")

#         # Demo dataset buttons
#         demo_btns = {}
#         with gr.Row():
#             for dataset_name in DEMO_DATASETS:
#                 demo_btn = gr.Button(f"{DEMO_DATASETS[dataset_name]['name']} Demo")
#                 demo_btns[dataset_name] = demo_btn

#             for name, demo_btn in demo_btns.items():
#                 # Set up the event chain for each demo button
#                 print(name, demo_btn)
#                 demo_btn.click(
#                     fn=disable_all_inputs,  # First disable all inputs
#                     inputs=[
#                         gr.Textbox(value=name, visible=False),
#                         chatbot,
#                         demo_btn,
#                         download_btn,
#                         msg,
#                         gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)  # Pass number of buttons instead
#                     ],
#                     outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
#                     queue=True
#                 ).then(
#                     fn=load_demo_dataset,
#                     inputs=[gr.Textbox(value=name, visible=False), chatbot, demo_btn, download_btn],
#                     outputs=[chatbot, demo_btn, download_btn, msg],
#                     queue=True,
#                     concurrency_limit=MAX_CONCURRENT_REQUESTS
#                 ).then(
#                     fn=process_message,
#                     inputs=[msg, chatbot, download_btn],
#                     outputs=[chatbot, download_btn],
#                     queue=True,
#                     concurrency_limit=MAX_CONCURRENT_REQUESTS
#                 ).then(
#                     fn=enable_all_inputs,
#                     inputs=[gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)],
#                     outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
#                     queue=True
#                 ).then(
#                     fn=lambda: "",
#                     outputs=[msg]
#                 )

#         # Event handlers with queue enabled
#         msg.submit(
#             fn=disable_all_inputs,  # First disable all inputs
#             inputs=[
#                 gr.Textbox(value="", visible=False),
#                 chatbot,
#                 gr.Button(visible=False),
#                 download_btn,
#                 msg,
#                 gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)  # Pass number of buttons instead
#             ],
#             outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
#             queue=True
#         ).then(
#             fn=process_message,
#             inputs=[msg, chatbot, download_btn],
#             outputs=[chatbot, download_btn],
#             concurrency_limit=MAX_CONCURRENT_REQUESTS,
#             queue=True
#         ).then(
#             fn=enable_all_inputs,
#             inputs=[gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)],
#             outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
#             queue=True
#         ).then(
#             fn=lambda: "",
#             outputs=[msg]
#         )

#         reset_btn.click(
#             fn=clear_chat,
#             outputs=[chatbot],
#             queue=False  # No need for queue on reset
#         )

#         file_upload.upload(
#             fn=disable_all_inputs,  # First disable all inputs
#             inputs=[
#                 gr.Textbox(value="", visible=False),
#                 chatbot,
#                 gr.Button(visible=False),
#                 download_btn,
#                 msg,
#                 gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)  # Pass number of buttons instead
#             ],
#             outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
#             queue=True
#         ).then(
#             fn=handle_file_upload,
#             inputs=[file_upload, chatbot, file_upload, download_btn],
#             outputs=[chatbot, file_upload, download_btn],
#             concurrency_limit=MAX_CONCURRENT_REQUESTS,
#             queue=True
#         ).then(
#             fn=enable_all_inputs,
#             inputs=[gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)],
#             outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
#             queue=True
#         )

#         # Download report handler with updated visibility
#         download_btn.click()

# if __name__ == "__main__":
#     demo.queue(default_concurrency_limit=MAX_CONCURRENT_REQUESTS,
#                max_size=MAX_QUEUE_SIZE)  # Enable queuing at the app level
#     demo.launch(share=True)

from datetime import datetime
import json
import asyncio
import time
from pydantic import BaseModel
import websockets
import multiprocessing



WS_URL = "ws://localhost:8000/ws_server"

websocket = None

async def work(data):
    print(data)
    await websocket.send('ok.')

async def ws_connect(s_id):
    url = f"{WS_URL}/{s_id}"
    print('url:', url)
    websocket = await websockets.connect(url)
    await websocket.send(f"hello, i'm {s_id}")
    msg = await websocket.recv()
    print(f"Recived from server: {msg}")
    await websocket.send('ok')
    while True:
        msg = await websocket.recv()
        print(f"Recived from server0: {msg}")
        data = json.loads(msg)
        print('data:', data)
        workflow = multiprocessing.Process(target=work, args=(data,))
        workflow.start()

    # async with websockets.connect(url) as websocket:
    #     print(f"connect to {url} ok.")
    #     #
    #     while True:
    #         msg = await websocket.recv()
    #         print(f"Recived from server: {msg}")
    #         data = json.loads(msg)


# async def main():
#     await ws_connect("server001")

# if __name__ == "__main__":
#     asyncio.run(main())
#     pass

import os
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, Response, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
from Gradio.demo_config import get_demo_config
from global_setting.Initialize_state import global_state_initialization
from preprocess.stat_info_functions import convert_stat_info_to_text, stat_info_collection
from preprocess.dataset import knowledge_info
from preprocess.eda_generation import EDA
from algorithm.filter import Filter
from algorithm.rerank import Reranker
from algorithm.program import Programming
from postprocess.judge import Judge
from postprocess.report_generation import Report_generation
from postprocess.visualization import Visualization


# from pydantic import BaseSettings

# settings
# class Settings(BaseSettings):


load_dotenv()

# path
# UPLOAD_FOLDER = "./demo_data"
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
# DOWNLOAD_FOLDER = './'
DOWNLOAD_FOLDER = os.getenv('DOWNLOAD_FOLDER')

# openai

OPENAI_OGANIZATION = os.getenv('OPENAI_OGANIZATION')
OPENAI_PROJECT = os.getenv('OPENAI_PROJECT')
OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')


app = FastAPI()

origins = [
    '*',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/demo_data", StaticFiles(directory=UPLOAD_FOLDER), name="demo_data")


@app.websocket("/test")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    target_path = ""
    DEMO_DATASETS["Abalone"][""]
    while True:
        u = []
        for y in test_yield(u):
            print(y, u)
            await websocket.send_json({'y':y})
            # await websocket.send_json(f"{'y':{y}}")
            # await websocket.send_text(f"y:{y}")
            res = await websocket.receive_text()
            print('res:', res)
            time.sleep(1) 
        break

    # websocket.close()


def test_yield(a):
    x = -1
    for i in range(10):
        yield x
        x += 2
        a.append(x)
    return x

async def sendMessageWithProcessing(websocket: WebSocket, role: str, messages: str):
    response_data = {
        'processing': True,
        'disable_btn': True,
        'data': {
            'role': role,
            'messages': messages
        }
    } 
    await websocket.send_json(response_data)


@app.websocket("/workflow")
async def websocket_workflow(websocket: WebSocket):
    await websocket.accept()

    chat_history = []
    target_path = None
    output_dir = None
    REQUIRED_INFO = {
        'data_upload': False,
        'initial_query': False,
    }
    MAX_CONCURRENT_REQUESTS = 5
    MAX_QUEUE_SIZE = 10

    # Abalone
    target_path = DEMO_DATASETS["Abalone"]["path"]

    # res data
    response_data = {
        'processing': False,
        'disable_btn': False,
        'data': {
            # chatbot, user 
            'role': 'chatbot',
            # image?
            'message': ""
        }
    }

    while True:
        #  
        """
            data
            {
                message:
                upload_file:
                demo: 

            }
        """
        data = await websocket.receive_json()
        # Update states:
        print('receive data:', data)
        message = data['message']
        REQUIRED_INFO['initial_query'] = message


        if data['demo']:
            demo_name = data['demo']
            if demo_name in DEMO_DATASETS:
                demo_dataset = DEMO_DATASETS[demo_name]
                source_path = demo_dataset["path"]
                date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', '')),
                            exist_ok=True)

                target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''),
                                        os.path.basename(source_path))
                output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''))
                shutil.copy(source_path, target_path)


                REQUIRED_INFO['data_uploaded'] = True
                REQUIRED_INFO['initial_query'] = True

                message = demo_dataset["query"]
                # user message
                response_data = {
                    "processing": True,
                    "disable_btn": True,
                    "data": {
                        "role": 'user',
                        "messages": [{
                            "type": "text",
                            "content": message
                        }]
                    }
                }
                await websocket.send_json(response_data) 
                res = await websocket.receive_text()

                df = pd.read_csv(target_path)
                # chatbot message
                response_data = {
                    "processing": True,
                    "disable_btn": True,
                    "data": {
                        "role": 'chatbot',
                        "messages": [{
                            "type": "text",
                            "content": f"✅ Loaded demo dataset '{demo_name}' with {len(df)} rows and {len(df.columns)} columns."
                        }]
                    }
                }
                await websocket.send_json(response_data) 
                res = await websocket.receive_text()
                print('ok.')
            else:
                print('no demo.')
                response_data = {
                    "processing": False,
                    "disable_btn": False,
                    "data": {
                        "role": 'chatbot',
                        "messages": [{
                            "type": "text",
                            "content": f"❌ The demo dataset '{demo_name}' is not exists!!!"
                        }]
                    }
                }
                await websocket.send_json(response_data) 
                continue
        # data uploaded stat/
        elif data["upload_file"]:
            if os.path.isfile(data["upload_file"]):
                REQUIRED_INFO["data_uploaded"] = True
                target_path = data["upload_file"]
                output_dir = os.path.dirname(target_path)
                print('output_dir:', output_dir)

        # elif not data["upload_file"]:
            # pass

        # for status, msg in _workflow(REQUIRED_INFO, data['message']):
        #     print('res:', status, msg)
        print('ok1.')

        # check data_uploaded
        if not REQUIRED_INFO["data_uploaded"]:
            print('data_uploaded.')
            response_data = {
                'processing': False,
                'disable_btn': False,
                'data': {
                    # chatbot, user 
                    'role': 'chatbot',
                    # image?
                    'messages': [{
                        "type": "text",
                        "content": "Please upload your dataset first before proceeding."
                    }]
                }
            }
            await websocket.send_json(response_data)
            continue
        
        # check initial_query
        if not REQUIRED_INFO["initial_query"]:
            print('initial_query')
            response_data = {
                "processing": False,
                "disable_btn": False,
                "data": {
                    "role": 'chatbot',
                    "messages": [{
                        "type": "text",
                        "content": "Please input your initial query."
                    }]
                }
            }
            await websocket.send_json(response_data)
            continue

        try:
            # Initialize config and global state
            print('start:')
            config = get_demo_config()
            config.data_file = target_path
            config.initial_query = message

            args = type('Args', (), {})()
            for key, value in config.__dict__.items():
                setattr(args, key, value)
            
            # OpenAI 
            # args.organization = ""
            args.organization = OPENAI_OGANIZATION
            # args.project = ""
            args.project = OPENAI_PROJECT
            # args.apikey = None
            args.apikey = OPENAI_APIKEY

            if 'YES' in message:
                args.data_mode = 'real'
            elif 'NO' in message:
                args.data_mode = 'simulated'
            else:
                print('not feature indicator')
                response_data = {
                    "processing": False,
                    "disable_btn": False,
                    "data": {
                        "role": 'chatbot',
                        "messages": [{
                            "type": "text",
                            "content": "Please indicate if your dataset has meaningful feature names using 'YES' or 'NO', which would help us generate appropriate report for you."
                        }]
                    }
                }
                await websocket.send_json(response_data)
                continue

            response_data = {
                "processing": True,
                "disable_btn": True,
                "data": None
                # "data": {
                #     "role": "",
                #     "message": ""
                # }
            }
            await websocket.send_json(response_data)
            res = await websocket.receive_text()
            print('res:', res)
            

            
            # Add user message
            global_state = global_state_initialization(args)

            # Load data
            global_state.user_data.raw_data = pd.read_csv(target_path)
            global_state.user_data.processed_data = global_state.user_data.raw_data

            # yield
            response_data = {
                "processing": True,
                "disable_btn": True,
                "data": {
                    "role": "user",
                    "messages": [{
                        "type": "text",
                        "content": f"📈 Run statistical analysis on Dataset {target_path.split('/')[-1].replace('.csv', '')}..."
                    }]
                }
            }
            await websocket.send_json(response_data)
            res = await websocket.receive_text()
            print('res:', res)

            user_linear = global_state.statistics.linearity
            user_gaussian = global_state.statistics.gaussian_error

            global_state = stat_info_collection(global_state)
            global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

            if global_state.statistics.data_type == "Continuous":
                if user_linear is None:
                    # chat_history.append(("✍️ Generate residuals plots ...", None))
                    # yield chat_history, download_btn
                    response_data = {
                        "processing": True,
                        "disable_btn": True,
                        "data": {
                            "role": "user",
                            "messages": [{
                                "type": "text",
                                "content": "✍️ Generate residuals plots ..."
                            }]
                        }
                    }
                    await websocket.send_json(response_data)
                    res = await websocket.receive_text()
                    print('res:', res)
                    # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/residuals_plot.jpg',)))
                    # yield chat_history, download_btn
                    response_data = {
                        "processing": True,
                        "disable_btn": True,
                        "data": {
                            "role": "chatbot",
                            "messages": [{ 
                                "type": "image",
                                "content": f'{global_state.user_data.output_graph_dir}/residuals_plot.jpg'
                            }]
                        }
                    }
                    await websocket.send_json(response_data)
                    res = await websocket.receive_text()
                    print('res:', res)
                if user_gaussian is None:
                    # chat_history.append(("✍️ Generate Q-Q plots ...", None))
                    # yield chat_history, download_btn
                    response_data = {
                        "processing": True,
                        "disable_btn": True,
                        "data": {
                            "role": "user",
                            "messages": [{
                                "type": "text",
                                "content": "✍️ Generate Q-Q plots ..."
                            }]
                        }
                    }
                    await websocket.send_json(response_data)
                    res = await websocket.receive_text()
                    print('res:', res)
                    # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/qq_plot.jpg',)))
                    # yield chat_history, download_btn
                    response_data = {
                        "processing": True,
                        "disable_btn": True,
                        "data": {
                            "role": "chatbot",
                            "messages": [{ 
                                "type": "image",
                                "content": f'{global_state.user_data.output_graph_dir}/qq_plot.jpg'
                            }]
                        }
                    }
                    await websocket.send_json(response_data)
                    res = await websocket.receive_text()
                    print('res:', res)

            # chat_history.append((None, global_state.statistics.description))
            # yield chat_history, download_btn
            response_data = {
                "processing": True,
                "disable_btn": True,
                "data": {
                    "role": "chatbot",
                    "messages": [{
                        "type": "text",
                        "content": global_state.statistics.description
                    }]
                }
            }
            await websocket.send_json(response_data)
            res = await websocket.receive_text()
            print('res:', res)

            # Knowledge generation
            if args.data_mode == 'real':
                # chat_history.append(("🌍 Generate background knowledge based on the dataset you provided...", None))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="user",
                    messages=[{
                        "type": "text",
                        "content": "🌍 Generate background knowledge based on the dataset you provided..."
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)

                global_state = knowledge_info(args, global_state)

                knowledge_clean = str(global_state.user_data.knowledge_docs).replace("[", "").replace("]", "").replace('"',
                                                                                                                    "").replace(
                    "\\n\\n", "\n\n").replace("\\n", "\n")
                # chat_history.append((None, knowledge_clean))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="chatbot",
                    messages= [{
                        "type": "text",
                        "content": knowledge_clean
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)
            elif args.data_mode == 'simulated':
                global_state = knowledge_info(args, global_state)

            # EDA Generation
            # chat_history.append(("🔍 Run exploratory data analysis...", None))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="user",
                messages= [{
                    "type": "text",
                    "content": "🔍 Run exploratory data analysis..."
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            my_eda = EDA(global_state)
            my_eda.generate_eda()
            # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_corr.jpg',)))
            # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_dist.jpg',)))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages=[{
                        "type": "image",
                        "content": f'{global_state.user_data.output_graph_dir}/eda_corr.jpg'
                    },{
                        "type": "image",
                        "content": f'{global_state.user_data.output_graph_dir}/eda_dist.jpg'
                    }
                ]
            )
            res = await websocket.receive_text()
            print('res:', res)

            # Algorithm Selection
            if global_state.algorithm.selected_algorithm is None:
                # chat_history.append(("🤖 Select optimal causal discovery algorithm and its hyperparameter...", None))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="user",
                    messages=[{
                        "type": "text",
                        "content": "🤖 Select optimal causal discovery algorithm and its hyperparameter..."
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)
                filter = Filter(args)
                global_state = filter.forward(global_state)
                reranker = Reranker(args)
                global_state = reranker.forward(global_state)
                # chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
                # chat_history.append((None, f"🤔 Algorithm selection reasoning: {global_state.algorithm.selected_reason}"))
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="chatbot",
                    messages=[{
                            "type": "text",
                            "content": f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"
                        },{
                            "type": "text",
                            "content": f"🤔 Algorithm selection reasoning: {global_state.algorithm.selected_reason}"
                        }
                    ]
                )
                res = await websocket.receive_text()
                print('res:', res)
            else:
                # chat_history.append(
                #     ("🤖 Select optimal hyperparameter for your selected causal discovery algorithm...", None))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="user",
                    messages=[{
                        "type": "text",
                        "content": "🤖 Select optimal hyperparameter for your selected causal discovery algorithm..."
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)

                filter = Filter(args)
                global_state = filter.forward(global_state)
                reranker = Reranker(args)
                global_state = reranker.forward(global_state)
                # chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="user",
                    messages=[{
                        "type": "text",
                        "content": f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)

            hyperparameter_text = ""
            for param, details in global_state.algorithm.algorithm_arguments_json['hyperparameters'].items():
                value = details['value']
                explanation = details['explanation']
                hyperparameter_text += f"  Parameter: {param}\n"
                hyperparameter_text += f"  Value: {value}\n"
                hyperparameter_text += f"  Explanation: {explanation}\n\n"
            # chat_history.append(
            #     (None,
            #     f"📖 Hyperparameters for the selected algorithm {global_state.algorithm.selected_algorithm}: \n\n {hyperparameter_text}"))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages=[{
                    "type": "text",
                    "content": f"📖 Hyperparameters for the selected algorithm {global_state.algorithm.selected_algorithm}: \n\n {hyperparameter_text}"
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            # Causal Discovery
            # chat_history.append(("🔄 Run causal discovery algorithm...", None))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="user",
                messages=[{
                    "type": "text",
                    "content": "🔄 Run causal discovery algorithm..."
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            programmer = Programming(args)
            global_state = programmer.forward(global_state)
            # Visualization for Initial Graph
            # chat_history.append(("📊 Generate causal graph visualization...", None))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="user",
                messages=[{
                    "type": "text",
                    "content": "📊 Generate causal graph visualization..."
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)
            my_visual_initial = Visualization(global_state)
            pos = my_visual_initial.get_pos(global_state.results.raw_result)
            if global_state.user_data.ground_truth is not None:
                my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.jpg', pos)
                my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', pos)
                # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/true_graph.jpg',)))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="chatbot",
                    messages=[{
                        "type": "image",
                        "content": f'{global_state.user_data.output_graph_dir}/true_graph.jpg'
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)
            if global_state.results.raw_result is not None:
                my_visual_initial.plot_pdag(global_state.results.raw_result, 'initial_graph.jpg', pos)
                my_visual_initial.plot_pdag(global_state.results.raw_result, 'initial_graph.pdf', pos)
                # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/initial_graph.jpg',)))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="chatbot",
                    messages=[{
                        "type": "image",
                        "content": f'{global_state.user_data.output_graph_dir}/initial_graph.jpg'
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)

                my_report = Report_generation(global_state, args)
                global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
                print('graph analysis', global_state.logging.graph_conversion['initial_graph_analysis'])
                # chat_history.append((None, global_state.logging.graph_conversion['initial_graph_analysis']))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="chatbot",
                    messages=[{
                        "type": "text",
                        "content": global_state.logging.graph_conversion['initial_graph_analysis']
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)

            # Evaluation for Initial Graph
            
            # chat_history.append(("📝 Evaluate and Revise the initial result...", None))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages=[{
                    "type": "text",
                    "content": "📝 Evaluate and Revise the initial result..."
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            try:
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state)
            except Exception as e:
                print('error during judging:', e)
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state)
            my_visual_revise = Visualization(global_state)
            if args.data_mode=='real':
                # Plot Revised Graph
                if global_state.results.revised_graph is not None:
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.pdf', pos)
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.jpg', pos)
                    # chat_history.append((None, f"This is the revised graph with Bootstrap and LLM techniques"))
                    # yield chat_history, download_btn
                    await sendMessageWithProcessing(
                        websocket=websocket,
                        role="chatbot",
                        messages=[{
                            "type": "text",
                            "content": f"This is the revised graph with Bootstrap and LLM techniques"
                        }]
                    )
                    res = await websocket.receive_text()
                    print('res:', res)
                    # chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/revised_graph.jpg',)))
                    # yield chat_history, download_btn
                    await sendMessageWithProcessing(
                        websocket=websocket,
                        role="chatbot",
                        messages=[{
                            "type": "image",
                            "content": f'{global_state.user_data.output_graph_dir}/revised_graph.jpg'
                        }]
                    )
                    res = await websocket.receive_text()
                    print('res:', res)
            # Plot Bootstrap Heatmap
            paths = my_visual_revise.boot_heatmap_plot()
            # chat_history.append(
            #     (None, f"The following heatmaps show the confidence probability we have on different kinds of edges"))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages=[{
                    "type": "text",
                    "content": f"The following heatmaps show the confidence probability we have on different kinds of edges"
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            messages = []
            for path in paths:
                # chat_history.append((None, (path,)))
                # yield chat_history, download_btn
                messages.append({
                    "type": "image",
                    "content": path
                })
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages=messages
            )
            res = await websocket.receive_text()
            print('res:', res)

            # chat_history.append((None, "✅ Causal discovery analysis completed"))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages= [{
                    "type": "text",
                    "content": "✅ Causal discovery analysis completed"
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            # Report Generation
            # chat_history.append(("📝 Generate comprehensive report and it may take a few minutes, stay tuned...", None))
            # yield chat_history, download_btn
            await sendMessageWithProcessing(
                websocket=websocket,
                role="user",
                messages=[{
                    "type": "text",
                    "content": "📝 Generate comprehensive report and it may take a few minutes, stay tuned..."
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            report_gen = Report_generation(global_state, args)
            report = report_gen.generation(debug=False)
            report_gen.save_report(report)
            report_path = os.path.join(output_dir, 'output_report', 'report.pdf')
            while not os.path.isfile(report_path):
                # chat_history.append((None,
                #                     "❌ An error occurred during the Report Generation, we are trying again and please wait for a few minutes."))
                # yield chat_history, download_btn
                await sendMessageWithProcessing(
                    websocket=websocket,
                    role="chatbot",
                    messages=[{
                        "type": "text",
                        "content": "❌ An error occurred during the Report Generation, we are trying again and please wait for a few minutes."
                    }]
                )
                res = await websocket.receive_text()
                print('res:', res)

                report_gen = Report_generation(global_state, args)
                report = report_gen.generation(debug=False)
                report_gen.save_report(report)

            # Final steps
            # chat_history.append((None, "🎉 Analysis complete!"))
            # chat_history.append((None, "📥 You can now download your detailed report using the download button below."))
            await sendMessageWithProcessing(
                websocket=websocket,
                role="chatbot",
                messages=[{
                        "type":"text",
                        "content": "🎉 Analysis complete!"
                    },{
                        "type": "text",
                        "content": "📥 You can now download your detailed report using the download button below."
                }]
            )
            res = await websocket.receive_text()
            print('res:', res)

            REQUIRED_INFO['processing'] = False

            # download_btn = gr.DownloadButton(
            #     "📥 Download Exclusive Report",
            #     size="sm",
            #     elem_classes=["icon-button"],
            #     scale=1,
            #     value=os.path.join(output_dir, 'output_report', 'report.pdf'),
            #     interactive=True
            # )
            # yield chat_history, download_btn
            output_report=os.path.join(output_dir, 'output_report', 'report.pdf'),
            
            response_data = {
                "processing": False,
                "disable_btn": False,
                # "data": None,
                "data": {
                    "role": 'chatbot',
                    "messages": [{
                        "type": "file",
                        "content": output_report
                    }]
                },
                "output_report": output_report
            }
            await websocket.send_json(response_data)
            res = await websocket.receive_text()
            print('res:', res)
            # chat_history.append((None, ""))
            # return chat_history, download_btn
            # TODO:
            continue

        except Exception as err:
            print(str(err))
            import traceback
            traceback.print_exc()
            response_data = {
                "processing": False,
                "disable_btn": False,
                "data": {
                    "role": 'chatbot',
                    "messages": [{
                        "type": "text",
                        "content": f"❌ An error occurred during analysis: {str(err)}, please try again"
                    }]
                }
            }
            await websocket.send_json(response_data)
            res = await websocket.receive_text()
            print('res:', res)
            continue

   
async def _workflow(REQUIRED_INFO, message):
    """
    return:
        status:
        msg:
        report:
        
    """
    print('REQUIRED_INFO:', REQUIRED_INFO)

    if not REQUIRED_INFO["data_upload"]:
        print('not uploaded')
        status = 'error'
        msg = ''
        yield status, msg
        
    if not REQUIRED_INFO['initial_query']:
        pass


    try:
        # Initialize config and global state

        # config

        # args
        args = type('Args', (), {})()

        # message
        if 'YES' in message:
            args.data_mode = 'real'
        elif 'NO' in message:
            args.data_mode = 'simulated'
        else:
            print('not feature indicator')
            status = 'no indicator'
            msg = "Please indicate if your dataset has meaningful feature names using 'YES' or 'NO', which would help us generate appropriate report for you."


        pass
    except Exception as e:

        status = 'error'
        msg = str(e)

        print(str(e))
        yield status, msg
        # return status, msg


@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    """
    使用UploadFile类的优势:
    1.文件开始存储在内存中，使用内存达到阈值后，将被保存在磁盘中
    2.适合于图片、视频大文件
    3.可以获取上传的文件的元数据，如文件名，创建时间等
    4.有文件对象的异步接口
    5.上传的文件是Python文件对象,可以使用write()、read()、seek()、close()等操做
    :param file:
    :return:
    """

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(
        os.path.join(
            UPLOAD_FOLDER, 
            date_time, 
            os.path.basename(file.filename).replace('.csv', '')
        ), 
        exist_ok=True
    )
    target_path = os.path.join(
        UPLOAD_FOLDER,
        date_time,
        os.path.basename(file.filename).replace('.csv', ''),
        os.path.basename(file.filename)
    )
    output_dir = os.path.join(
        UPLOAD_FOLDER,
        date_time,
        os.path.basename(file.filename).replace('.csv', '')
    )
    with open(f"{target_path}", 'wb') as f:
        for i in iter(lambda: file.file.read(1024 * 1024 *10), b''):
            f.write(i)
    f.close()
    return { 
        "file_name": file.filename,
        "file_path": target_path
    }



class DownloadFilePostData(BaseModel):
    filepath: str

# @app.get("/download_file/{filepath}")
# def download_file(filepath: str):
@app.post("/download_file/")
def download_file(data: DownloadFilePostData):
    print('filepath:', data.filepath)
    # file_path = os.path.join('./', data.filepath)
    file_path = os.path.join(DOWNLOAD_FOLDER, data.filepath)
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            contents = file.read()
        return Response(content=contents, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename=example.txt"})
    return {"message": "文件未找到"}

@app.get("/test/")
def test():
    return {"msg": "hello world. ^_^"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_workflow_v1:app", reload=True, port=8000)