import os
import json
import re



def format_time_series_in_text(text):
    """Find time series arrays in text and format them for better display"""
    # Pattern to find array-like structures [x, y, z, ...]
    pattern = r'\[([0-9.,\s-]+)\]'
    
    def format_match(match):
        # Get the content of the array
        array_content = match.group(1)
        
        # Format it with line breaks after every 8-10 elements
        elements = array_content.split(',')
        formatted_elements = []
        chunk_size = 8
        
        for i in range(0, len(elements), chunk_size):
            chunk = elements[i:i+chunk_size]
            formatted_elements.append(', '.join(chunk))
        
        formatted_array = ',<br>'.join(formatted_elements)
        # Wrap in a span with special class for formatting
        return f'<span class="time-series">[{formatted_array}]</span>'
    
    # Replace all occurrences of arrays in the text
    formatted_text = re.sub(pattern, format_match, text)
    return formatted_text

def clean_option_text(option_text):
    """Clean and format option text for better display"""
    # If it's a list representation, format it nicely
    if option_text.startswith('[') and option_text.endswith(']'):
        try:
            # Try to parse it as JSON
            data = json.loads(option_text)
            if isinstance(data, list):
                # Format with limited precision for floats
                formatted = [f"{float(x):.2f}" if isinstance(x, float) else str(x) for x in data]
                # Show all values, properly formatted
                display = ", ".join(formatted)
                return display
        except:
            pass
    return option_text

def generate_mturk_html(questions):
    """Generate MTurk HTML from a list of question dictionaries"""
    
    # Start with the HTML template
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiple-Choice Questionnaire</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .instructions {
            background-color: #f9f9f9;
            border-left: 4px solid #0066cc;
            padding: 10px 15px;
            margin-bottom: 20px;
        }
        .question-container {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            background-color: #fff;
        }
        .question {
            font-weight: bold;
            margin-bottom: 15px;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .time-series {
            font-family: monospace;
            max-width: 100%;
            overflow-x: auto;
            background-color: #f9f9f9;
            padding: 5px;
            border-radius: 3px;
        }
        .options {
            margin-left: 10px;
        }
        .option {
            margin-bottom: 10px;
        }
        .option-content {
            display: inline-block;
            vertical-align: top;
            width: 90%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .numeric-array {
            max-width: 100%;
            overflow-x: auto;
            white-space: normal;
            padding: 5px 0;
        }
        input[type="radio"] {
            margin-right: 10px;
        }
        label {
            display: inline-block;
            cursor: pointer;
        }
        label:hover {
            color: #0066cc;
        }
        .submit-container {
            margin-top: 30px;
            text-align: center;
        }
        .submit-button {
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #0052a3;
        }
        .required {
            color: red;
        }
        .question-id {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <form id="mturk-questionnaire">
        <div class="instructions">
            <h2>Survey Instructions</h2>
            <p>Please answer all questions by selecting the option that you think is most appropriate. All questions are required to complete the HIT.</p>
        </div>
"""

    # Add each question
    for i, q in enumerate(questions, 1):
        question_id = q.get("Question ID", f"question_{i}")
        question_text = q.get("Question", "")
        question_type = q.get("Question Type", "")
        
        # Process the question text to format time series arrays for better display
        question_text = format_time_series_in_text(question_text)
        
        # Format the question section
        html += f"""
        <!-- Question {i} -->
        <div class="question-container">
            <div class="question-id">ID: {question_id}</div>
            <div class="question">
                <span class="required">*</span> Question {i}: {question_text}
            </div>
            <div class="options">
"""
        
        # Add each option
        for j in range(1, 5):
            option_key = f"Option {j}"
            if option_key in q:
                option_text = clean_option_text(q[option_key])
                # Check if this is a numeric array and apply special formatting
                is_numeric_array = option_text and ',' in option_text and any(c.isdigit() for c in option_text)
                content_class = "option-content numeric-array" if is_numeric_array else "option-content"
                
                html += f"""
                <div class="option">
                    <input type="radio" id="q{i}_option{j}" name="{question_id}" value="{j}" required>
                    <label for="q{i}_option{j}">
                        <div class="{content_class}">{option_text}</div>
                    </label>
                </div>
"""

        # Close the question div
        html += """
            </div>
        </div>
"""

    # Add the submit button and closing tags
    html += """
        <div class="submit-container">
            <button class="submit-button" type="submit">Submit</button>
        </div>
    </form>

    <script>
        document.getElementById('mturk-questionnaire').addEventListener('submit', function(event) {
            event.preventDefault();
            
            // Get all question names
            const questions = [];
            const radios = document.querySelectorAll('input[type="radio"]');
            radios.forEach(function(radio) {
                if (!questions.includes(radio.name)) {
                    questions.push(radio.name);
                }
            });
            
            // Check if all questions are answered
            let allAnswered = true;
            let unansweredQuestions = [];
            
            questions.forEach(function(question) {
                const options = document.getElementsByName(question);
                let questionAnswered = false;
                
                options.forEach(function(option) {
                    if (option.checked) {
                        questionAnswered = true;
                    }
                });
                
                if (!questionAnswered) {
                    allAnswered = false;
                    unansweredQuestions.push(question);
                }
            });
            
            if (!allAnswered) {
                alert('Please answer all questions before submitting. Unanswered questions: ' + unansweredQuestions.join(', '));
                return;
            }
            
            // Collect the answers for submission
            const answers = {};
            questions.forEach(function(question) {
                const options = document.getElementsByName(question);
                options.forEach(function(option) {
                    if (option.checked) {
                        answers[question] = option.value;
                    }
                });
            });
            
            console.log('Answers:', answers);
            alert('Thank you for completing the questionnaire!');
            
            // In a real MTurk HIT, you would use code like this:
            // document.getElementById('mturk-form').submit();
        });
    </script>
</body>
</html>
"""
    return html

def convert_json_to_mturk_html(json_file, output_file):
    """Convert JSON file with questions to MTurk HTML"""
    
    # Load the questions from JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    elif isinstance(data, list):
        questions = data
    else:
        questions = [data]  # Single question case
    
    # Remove "Ground Truth" from each question
    for q in questions:
        if "Ground Truth" in q:
            del q["Ground Truth"]
    
    # Generate the HTML
    html = generate_mturk_html(questions)
    
    # Write the output file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Successfully converted {len(questions)} questions to HTML format.")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    
    json_file = "/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/caption_retrieval_perturbed_questions.json"
    output_file = "/home/ubuntu/thesis/data/samples/new samples no overlap/hard_questions_small/caption_retrieval_perturbed_questions.html"
    convert_json_to_mturk_html(json_file, output_file)