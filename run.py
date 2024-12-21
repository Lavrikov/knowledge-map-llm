import gradio as gr
import sqlite3
from vllm import LLM, SamplingParams
import torch.multiprocessing as mp
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import re

# multiquestions per chunk
# gradio interface to look out of progress

def split_text(text, size=512):
    """
    simple text splitting
    """
    return [text[i:i + size] for i in range(0, len(text), size)]

def process_stream(Logs_id, model, tokenizer, sampling_params, cursor, topic_tag = "NOFILE"):
    """
    generate answer and send to gradio
    """
    cursor.execute("SELECT * FROM Logs_texts WHERE Logs_id = ?", (Logs_id,))
    logs_data = cursor.fetchone()
    if not logs_data:
        yield "Log file not found"
        return

    #logs_id = logs_data[1]
    logs_text = logs_data[1]

    # Get questions for topic
    cursor.execute("SELECT * FROM GeneralQuestions WHERE topic_tag = ?", (topic_tag,))
    question_rows = cursor.fetchall()

    chunks = split_text(logs_text)

    for question in question_rows:
        #print(question)
        question = question[2]
        for i, chunk in enumerate(chunks):
            prompt = """
            You are senior software developer trying to help junior colleague to debug errors. 
            You are reading logs with errors row by row. You are seeking relevant information to question.
            Do information relevant to question: {Question}? contains in: {Chunk}
            Choose statement from bellow:
            1. Yes, information relevant to question
            2. No, no relevant to question information
            3. Partially, some information appears in text
            """

            messages = [
                {
                    "role": "user", 
                    "content": prompt.format(
                            Question=question,
                            Chunk=chunk
                        )
                } 
            ] 
            text_m = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
            responses = model.generate(
                [text_m],
                sampling_params,
                use_tqdm=False
            )
            
            responses = [x.outputs[0].text for x in responses]

            yield f"**Question:** {question}\n\n**Chunk {i + 1}:**\n{responses[0]}\n\n**Text :**\n{chunk}"


def main():

    # Connect to database
    connection = sqlite3.connect("./data/log_database.db", check_same_thread=False)
    cursor = connection.cursor()

    # vLLM
    model_path = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

    llm = LLM(
       model_path,
       quantization="gptq",
       tensor_parallel_size=1,
       gpu_memory_utilization=0.90,
       trust_remote_code=True,
       dtype="half",
       enforce_eager=True,
       max_model_len=512,
       disable_log_stats=True
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=1,
        top_k=1,
        temperature=0,
        seed=777, # Seed
        skip_special_tokens=False,
        max_tokens=1,
        logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["1", "2", "3"])]
    )

    # Get logs text to analysis
    cursor.execute("SELECT * FROM Logs_texts")
    logs_list = cursor.fetchall()
    logs_options = [f"ID: {row[0]}" for row in logs_list]

    # Gradio to out results in the web
    with gr.Blocks() as demo:
        gr.Markdown("### Answer generation with LLM")
        selected_logs = gr.Dropdown(logs_options, label="Choose log file")
        generate_button = gr.Button("Run analysis")


        responses_output = gr.Textbox(label="Answers for questions", lines=20)
        relevant_chunks_display = gr.Textbox(label=" Found related information (answer 1 yes or 3 partially yes)", lines=10)

        def generate(selected_option):
            logs_id = int(selected_option.split(" ")[1])
            accumulated_chunks = ""
            current_question = ""
            responses_content = ""

            for partial_response in process_stream(logs_id, llm, tokenizer, sampling_params, cursor):
                # Select current question
                if "**Question:**" in partial_response:
                    current_question = partial_response.split("**Question:**")[1].split("\n\n")[0].strip()
                    print("current_question", current_question)

                # Process chunk tetx and answer
                if "**Chunk" in partial_response:
                    print("partial_response", partial_response)
                    response_line = partial_response.split("\n")[3].strip()
                    chunk = partial_response.split("**Chunk")[1].split(":**\n")#[1].strip()
                    answer_code =  chunk[1]
                    related_text = chunk[2]
                    print("response_line", response_line)

                    if response_line.startswith("1") or response_line.startswith("3"):
                        accumulated_chunks += f"-----{current_question}------?\n answer -{answer_code}-\n\n{related_text}\n\n##########################\n\n"
                        print("accumulated_chunks",accumulated_chunks)

                # Update view
                responses_content += partial_response + "\n\n"
                print("#################################################")
                yield responses_content, accumulated_chunks

        # connect button with function
        generate_button.click(
            generate,
            inputs=selected_logs,
            outputs=[responses_output, relevant_chunks_display]
        )

        # run gradio
        demo.launch(share=True)

if __name__ == "__main__":
    main()
