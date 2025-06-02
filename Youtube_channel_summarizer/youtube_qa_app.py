import gradio as gr
import yt_dlp
import pandas as pd
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi as yta
from urllib.parse import urlparse, parse_qs
import time
import os
import tempfile

class YouTubeChannelQA:
    def __init__(self):
        self.transcripts_dict = {}
        self.df = None
        self.model = None
        self.chat_session = None
        self.processed_text = ""
        
    def extract_video_id(self, url):
        """Extract video ID from a YouTube URL"""
        try:
            return parse_qs(urlparse(url).query)['v'][0]
        except:
            return None
    
    def fetch_channel_videos(self, channel_url, max_videos=None):
        """Fetch video URLs from a YouTube channel"""
        try:
            ydl_opts = {
                'extract_flat': True,
                'quiet': True,
                'skip_download': True,
            }
            
            # Only add playlistend if max_videos is specified and not "all"
            if max_videos is not None and max_videos > 0:
                ydl_opts['playlistend'] = max_videos
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                video_urls = [entry['url'] for entry in info['entries'] if 'url' in entry]
                return video_urls
        except Exception as e:
            raise Exception(f"Error fetching channel videos: {str(e)}")
    
    def fetch_transcripts(self, video_urls, progress_callback=None):
        """Fetch transcripts for all videos with retry logic"""
        transcripts_dict = {}
        total_videos = len(video_urls)
        max_attempts = 10
        
        # Initial attempt to fetch all transcripts
        for i, url in enumerate(video_urls, start=1):
            try:
                vid_id = self.extract_video_id(url)
                if not vid_id:
                    continue
                    
                data = yta.get_transcript(vid_id)
                transcript = ''
                
                for value in data:
                    if 'text' in value:
                        transcript += value['text'] + ' '
                
                final_output = transcript.replace('\n', ' ').strip()
                transcripts_dict[i] = final_output
                
                if progress_callback:
                    progress_callback(f"Processed {len(transcripts_dict)}/{total_videos} videos (Initial attempt)")
                    
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error with video {i} (will retry): {str(e)}")
                continue
        
        # Retry logic for missing transcripts
        attempt = 1
        while len(transcripts_dict) < total_videos and attempt <= max_attempts:
            if progress_callback:
                progress_callback(f"Retry attempt {attempt}/{max_attempts} - Have {len(transcripts_dict)}/{total_videos} transcripts")
            
            for i, url in enumerate(video_urls, start=1):
                if i in transcripts_dict:
                    continue  # Already fetched
                
                try:
                    vid_id = self.extract_video_id(url)
                    if not vid_id:
                        continue
                        
                    data = yta.get_transcript(vid_id)
                    transcript = ''
                    
                    for value in data:
                        if 'text' in value:
                            transcript += value['text'] + ' '
                    
                    final_output = transcript.replace('\n', ' ').strip()
                    transcripts_dict[i] = final_output
                    
                    if progress_callback:
                        progress_callback(f"Successfully fetched transcript for video {i} on retry {attempt}")
                        
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Retry {attempt} error for video {i}: {str(e)}")
                    continue
            
            attempt += 1
            time.sleep(1)  # Sleep between attempts to avoid hitting rate limits
        
        # Final status update
        if progress_callback:
            success_rate = len(transcripts_dict) / total_videos * 100 if total_videos > 0 else 0
            progress_callback(f"Final: {len(transcripts_dict)}/{total_videos} transcripts fetched ({success_rate:.1f}% success rate)")
        
        return transcripts_dict
    
    def summarize_text(self, text):
        """Summarize a single text using Gemini"""
        try:
            prompt = f"""Summarize the following text while maintaining the original style and level of detail.
            Be sure to preserve all specific information, including tools, approaches, company names, and other concrete details.
            The content will be used for learning purposes, so terminology and explanations are crucial.
            
            {text}"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {e}"
    
    def process_channel(self, channel_url, gemini_api_key, max_videos, progress=gr.Progress()):
        """Main processing function"""
        try:
            # Configure Gemini
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
            
            progress(0.1, desc="Fetching channel videos...")
            
            # Handle "all videos" option
            videos_to_fetch = None if max_videos == 0 else max_videos
            
            # Fetch video URLs
            video_urls = self.fetch_channel_videos(channel_url, videos_to_fetch)
            
            if not video_urls:
                return "No videos found in the channel.", "", gr.update(visible=False)
            
            videos_text = "all videos" if max_videos == 0 else f"{len(video_urls)} videos"
            progress(0.2, desc=f"Found {len(video_urls)} videos. Processing {videos_text} with retry logic...")
            
            # Fetch transcripts with retry logic
            def update_progress(msg):
                current_progress = 0.2 + 0.4 * (len(self.transcripts_dict) / len(video_urls))
                progress(current_progress, desc=msg)
            
            self.transcripts_dict = self.fetch_transcripts(video_urls, update_progress)
            
            if not self.transcripts_dict:
                return "No transcripts could be fetched after retry attempts.", "", gr.update(visible=False)
            
            progress(0.6, desc="Creating summaries...")
            
            # Create DataFrame and summarize
            self.df = pd.DataFrame(list(self.transcripts_dict.items()), columns=['index', 'text'])
            self.df['gemini_summary'] = self.df['text'].apply(self.summarize_text)
            
            progress(0.8, desc="Preparing chat system...")
            
            # Prepare text for chat
            self.processed_text = ' '.join(self.df['gemini_summary'].dropna().astype(str))
            
            # Initialize chat session
            system_prompt = f"""You are a helpful assistant. Answer questions based on the provided text that comes from specific Youtube channel's transcripts.
            If you don't know the answer, say 'I don't know'.
            
            The text is as follows:
            
            {self.processed_text}"""
            
            chat_model = genai.GenerativeModel('models/gemini-2.5-pro-preview-05-06')
            self.chat_session = chat_model.start_chat(history=[])
            self.chat_session.send_message(system_prompt)
            
            progress(1.0, desc="Processing complete!")
            
            # Generate summary report
            success_rate = len(self.transcripts_dict) / len(video_urls) * 100
            processing_mode = "All available videos" if max_videos == 0 else f"Limited to {max_videos} videos"
            summary_stats = f"""
            📊 **Processing Summary:**
            - Channel: {channel_url}
            - Processing mode: {processing_mode}
            - Videos found: {len(video_urls)}
            - Transcripts fetched: {len(self.transcripts_dict)}
            - Success rate: {success_rate:.1f}%
            - Total words: {sum(len(text.split()) for text in self.transcripts_dict.values()):,}
            
            ✅ Ready to answer questions!
            """
            
            return summary_stats, "", gr.update(visible=True)
            
        except Exception as e:
            return f"Error: {str(e)}", "", gr.update(visible=False)
    
    def chat_with_data(self, message, history):
        """Chat function for Gradio interface"""
        if not self.chat_session:
            return "Please process a YouTube channel first!"
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def export_data(self):
        """Export processed data to CSV"""
        if self.df is not None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            self.df.to_csv(temp_file.name, index=False)
            return temp_file.name
        return None

# Initialize the app
app = YouTubeChannelQA()

# Create Gradio interface
with gr.Blocks(title="YouTube Channel Q&A", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎥 YouTube Channel Q&A Assistant")
    gr.Markdown("Extract transcripts from YouTube channels and ask questions using AI!")
    
    with gr.Row():
        with gr.Column(scale=2):
            channel_url = gr.Textbox(
                label="YouTube Channel URL",
                placeholder="https://www.youtube.com/@channelname/videos",
                info="Enter the full YouTube channel URL"
            )
            
            gemini_api_key = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="Your Gemini API key",
                info="Get your API key from Google AI Studio"
            )
            
            max_videos = gr.Slider(
                minimum=0,
                maximum=1000,
                value=50,
                step=1,
                label="Maximum Videos to Process",
                info="Set to 0 to process ALL videos in the channel, or choose a specific number (1-1000)"
            )
            
            process_btn = gr.Button("🔄 Process Channel", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            status_output = gr.Markdown("👆 Fill in the details and click 'Process Channel' to start")
    
    # Chat interface (initially hidden)
    with gr.Group(visible=False) as chat_group:
        gr.Markdown("## 💬 Ask Questions")
        
        chatbot = gr.Chatbot(
            height=400,
            show_label=False,
            avatar_images=("👤", "🤖")
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about the channel content...",
                show_label=False,
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Sample questions
        with gr.Row():
            gr.Examples(
                examples=[
                    "What are the main topics covered in this channel?",
                    "Summarize the key insights from all videos",
                    "What trends can you identify in the content over time?",
                    "What are the company's main values and vision?"
                ],
                inputs=msg_input
            )
        
        # Export button
        with gr.Row():
            export_btn = gr.Button("📥 Export Data as CSV", variant="secondary")
            download_file = gr.File(label="Download", visible=False)
    
    # Event handlers
    process_btn.click(
        fn=app.process_channel,
        inputs=[channel_url, gemini_api_key, max_videos],
        outputs=[status_output, msg_input, chat_group]
    )
    
    def respond(message, history):
        if message.strip():
            bot_message = app.chat_with_data(message, history)
            history.append([message, bot_message])
        return history, ""
    
    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    def export_csv():
        file_path = app.export_data()
        if file_path:
            return gr.update(visible=True, value=file_path)
        return gr.update(visible=False)
    
    export_btn.click(
        fn=export_csv,
        outputs=download_file
    )

# Launch the app
if __name__ == "__main__":
    print("Starting YouTube Channel Q&A App...")
    print("If you can't access the local URL, try these alternatives:")
    print("1. http://localhost:7860")
    print("2. http://127.0.0.1:7860")
    print("3. Check your firewall settings")
    
    try:
        interface.launch(
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7860,         # Default Gradio port
            share=False,              # Set to True for public link
            debug=True,
            inbrowser=True            # Automatically open browser
        )
    except Exception as e:
        print(f"Error launching on port 7860: {e}")
        print("Trying alternative port 8080...")
        try:
            interface.launch(
                server_name="127.0.0.1",
                server_port=8080,
                share=False,
                debug=True,
                inbrowser=True
            )
        except Exception as e2:
            print(f"Error launching on port 8080: {e2}")
            print("Trying with share=True for public link...")
            interface.launch(
                share=True,  # This creates a public ngrok link
                debug=True
            )