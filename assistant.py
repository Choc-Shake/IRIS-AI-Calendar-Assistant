import os
import json
from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import ollama
import pytz

# ---------- Configuration ----------
GOOGLE_CREDENTIALS_FILE = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
SCOPES = ['https://www.googleapis.com/auth/calendar']
TIMEZONE = 'America/Edmonton'
MEMORY_FILE = 'memory.json'
OLLAMA_MODEL = "gemma3:4b"

# ---------- Date Helper ----------
def get_current_datetime():
    """Get current datetime with proper timezone"""
    return datetime.now(pytz.timezone(TIMEZONE))

# ---------- Google Calendar Helpers ----------
def authenticate_google_calendar():
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        print("❌ Error: credentials.json file not found.")
        return None
    creds = None
    token_file = 'token.json'
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    service = build('calendar', 'v3', credentials=creds)
    return service

def list_upcoming_events(service, max_results=10):
    """List upcoming events and return formatted string for the model"""
    now = get_current_datetime().isoformat()
    events_result = service.events().list(
        calendarId='primary', timeMin=now,
        maxResults=max_results, singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    
    if not events:
        return "No upcoming events found."
    
    # Format events in a natural way for the model to use
    formatted_events = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'No title')
        
        # Parse the datetime for better formatting
        try:
            if 'T' in start:
                dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = pytz.timezone(TIMEZONE).localize(dt)
                formatted_time = dt.strftime('%I:%M %p')
                formatted_date = dt.strftime('%B %d, %Y')
                formatted_events.append(f"• {formatted_time} on {formatted_date}: {summary}")
            else:
                # All-day event
                dt = datetime.fromisoformat(start)
                formatted_date = dt.strftime('%B %d, %Y')
                formatted_events.append(f"• All day on {formatted_date}: {summary}")
        except:
            formatted_events.append(f"• {start}: {summary}")
    
    return "\n".join(formatted_events)

def create_calendar_event(service, summary, start_time, end_time):
    event = {
        'summary': summary,
        'start': {'dateTime': start_time, 'timeZone': TIMEZONE},
        'end': {'dateTime': end_time, 'timeZone': TIMEZONE}
    }
    created_event = service.events().insert(calendarId='primary', body=event).execute()
    print(f"✅ Event created: {created_event.get('htmlLink')}")

def update_calendar_event(service, event_id, summary, start_time, end_time):
    event = service.events().get(calendarId='primary', eventId=event_id).execute()
    event['summary'] = summary
    event['start']['dateTime'] = start_time
    event['end']['dateTime'] = end_time
    updated_event = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
    print(f"✅ Event updated: {updated_event.get('htmlLink')}")

def delete_calendar_event(service, event_id):
    service.events().delete(calendarId='primary', eventId=event_id).execute()
    print(f"✅ Event deleted.")

def search_events(service, query):
    now = get_current_datetime().isoformat()
    events_result = service.events().list(
        calendarId='primary', timeMin=now,
        singleEvents=True, orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])
    return [e for e in events if query.lower() in e.get('summary', '').lower()]

# ---------- Ollama Helpers ----------
def query_ollama_full_context(memory, service=None):
    """
    Sends full conversation memory to Ollama using the chat API and returns parsed JSON response.
    """
    # Get current date for context
    current_date = get_current_datetime()
    current_year = current_date.year
    
    # Get actual events from calendar for context
    actual_events = list_upcoming_events(service) if service else "No calendar access"
    
    # Build system prompt with dynamic year handling
    system_prompt = f"""You are "IRIS", an intelligent calendar assistant created by Ishaan Abraham to help manage his Google Calendar alongside answering any other general query's he may have.

    ABOUT Ishaan:
    - Name: Ishaan Abraham
    - Born January 31, 2007
    - Born in Manama, Bahrain
    - Grew up in Mississauga, Ontario, Canada
    - Lives in Edmonton, Alberta, Canada
    - Currently in College

    YOUR IDENTITY:
    - Your name is IRIS
    - IRIS stands for Intelligent Response and Insight System
    - You are a personal Assistant for Ishaan Abraham
    - You should be friendly and helpful while being professional


TODAY'S DATE: {current_date.strftime('%B %d, %Y')}

ACTUAL CALENDAR EVENTS:
{actual_events}

IMPORTANT RULES:
1. When listing events, ONLY mention the actual events shown above. Do not make up events.
2. Format your response with each event on a new line using bullet points.
3. When the user mentions a date without specifying a year, determine the correct year based on the current date.
4. For dates: If the mentioned date has already passed this year, use next year. If it's in the future, use this year.
5. Respond to the name "IRIS" and be friendly, but be professional and slightly casual in your interactions.
6. Remember that you're talking to Ishaan Abraham and can use his name naturally in conversation.

[Add any other custom rules or behaviors here]

Based on the user's input, respond naturally and output JSON for any calendar action.
Your response should be in this exact JSON format:
{{
    "action": "create|update|delete|list|chat",
    "summary": "Event title",
    "start_time": "ISO datetime (e.g., 2025-09-02T11:00:00-06:00)",
    "end_time": "ISO datetime (e.g., 2025-09-02T14:00:00-06:00)",
    "reply": "Your natural language response"
}}

If the user asks to list events, use "action": "list" and format your reply naturally using ONLY the actual events above with each event on a new line.
If no calendar action is needed, use "action": "chat".
"""
    
    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in memory["conversation"]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
            options={"temperature": 0.1}
        )
        return parse_ollama_response(response["message"]["content"])
    except Exception as e:
        return {"action": "chat", "summary": "", "start_time": "", "end_time": "", "reply": f"Error: {str(e)}"}

def parse_ollama_response(raw_response):
    """
    Ensure Ollama response is parsed as dict with correct keys.
    """
    try:
        if isinstance(raw_response, dict):
            return raw_response
        
        # Extract JSON from response text
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            return json.loads(json_str)
        
        return {"action": "chat", "summary": "", "start_time": "", "end_time": "", "reply": raw_response}
    except Exception as e:
        return {"action": "chat", "summary": "", "start_time": "", "end_time": "", "reply": f"Parse error: {str(e)}"}

# ---------- Utilities ----------
def is_affirmative(user_input):
    affirmatives = ['yes', 'y', 'sure', 'affirmative', 'please do', 'that would be great', 'ok']
    return any(word in user_input.lower() for word in affirmatives)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump({"conversation": []}, f)
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

# ---------- Main ----------
def main():
    # Clear memory once on startup
    with open(MEMORY_FILE, "w") as f:
        json.dump({"conversation": [], "last_event": None}, f)

    service = authenticate_google_calendar()
    if not service:
        return

     # Enhanced welcome message
    print("🌙" + "="*60)
    print("IRIS Calendar Assistant initialized")
    print("="*60)
    print("Hello Ishaan! I'm IRIS, your intelligent calendar assistant.")
    print("I can help you manage your Google Calendar with natural conversation.")
    print("\nYou can:")
    print("• Schedule, Update, Delete, or List events")
    print("• Or just chat with me!")
    print("\nExamples:")
    print('  "Add meeting with Alex tomorrow at 2pm"')
    print('  "What do I have scheduled this week?"')
    print('  "Cancel my 3pm meeting today"')
    print("="*60)
    print("Type 'quit', 'exit', or 'bye' to end our session.")
    print("🌙" + "="*60)
    print()

    memory = load_memory()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
            print("IRIS: Goodbye Ishaan! Have a great day! 🌙")
            break

        # Add user input to conversation memory
        memory.setdefault("conversation", []).append({"role": "user", "content": user_input})

        # Query Ollama with full context (pass service for event listing)
        parsed = query_ollama_full_context(memory, service)

        # Add assistant response to memory
        memory["conversation"].append({"role": "assistant", "content": parsed.get("reply","")})
        save_memory(memory)

        # Print Ollama's natural language response
        print("IRIS:", parsed.get("reply",""))

        # Execute calendar actions if any
        action = parsed.get("action", "chat")
        details = parsed

        if action == "create":
            create_calendar_event(service, details['summary'], details['start_time'], details['end_time'])
            memory["last_event"] = details
            save_memory(memory)

        elif action == "update":
            last = memory.get("last_event")
            if last:
                events = search_events(service, last.get("summary", ""))
                if events:
                    target = events[0]  # For simplicity, take first match
                    update_calendar_event(service, target['id'], details['summary'], details['start_time'], details['end_time'])
                    memory["last_event"] = details
                    save_memory(memory)
                else:
                    print("No matching events found for update.")

        elif action == "delete":
            events = search_events(service, details.get("summary",""))
            if events:
                target = events[0]  # Take first match
                confirm = input(f"Do you want to delete '{target.get('summary')}'? (y/n): ")
                if confirm.lower() in ["y", "yes", "affirmative", "sure"]:
                    delete_calendar_event(service, target['id'])
                    memory["last_event"] = None
                    save_memory(memory)
            else:
                print("No matching events found for deletion.")

        elif action == "list":
            # Just let the model handle the response, we already provided the real events
            pass


if __name__ == '__main__':
    main()