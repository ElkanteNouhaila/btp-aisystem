import os
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def extract_attachments(payload):
    attachments = []

    def walk(parts):
        for part in parts:
            filename = part.get("filename")
            body = part.get("body", {})

            if filename and body.get("attachmentId"):
                attachments.append({
                    "filename": filename,
                    "attachment_id": body["attachmentId"],
                    "mimeType": part.get("mimeType")
                })

            # recursive case (VERY IMPORTANT)
            if "parts" in part:
                walk(part["parts"])

    if "parts" in payload:
        walk(payload["parts"])

    return attachments

def get_attachment(service, msg_id, attachment_id):
    attachment = service.users().messages().attachments().get(
        userId="me",
        messageId=msg_id,
        id=attachment_id
    ).execute()

    data = base64.urlsafe_b64decode(attachment["data"])
    return data

def authenticate_gmail():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    elif not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            SCOPES
        )

        creds = flow.run_local_server(
            port=8080,
            access_type="offline",
            prompt="consent"
        )

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def get_gmail_service():
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)
    return service



def extract_email_body(payload):
    body = ""

    def walk(parts):
        nonlocal body

        for part in parts:
            mime_type = part.get("mimeType")

            if mime_type == "text/plain":
                data = part.get("body", {}).get("data")
                if data:
                    body += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

            if "parts" in part:
                walk(part["parts"])

    if "parts" in payload:
        walk(payload["parts"])
    else:
        data = payload.get("body", {}).get("data")
        if data:
            body += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    return body

# def fetch_emails(max_results=10):
#     service = get_gmail_service()

#     results = service.users().messages().list(
#         userId="me",
#         maxResults=max_results
#     ).execute()

#     messages = results.get("messages", [])

#     emails = []

#     for msg in messages:
#         msg_data = service.users().messages().get(
#             userId="me",
#             id=msg["id"]
#         ).execute()

#         payload = msg_data["payload"]
#         headers = payload.get("headers", [])

#         # Subject
#         subject = next(
#             (h["value"] for h in headers if h["name"] == "Subject"),
#             "No Subject"
#         )

#         # Sender
#         sender = next(
#             (h["value"] for h in headers if h["name"] == "From"),
#             "Unknown Sender"
#         )

#         # Date
#         internal_date = msg_data.get("internalDate", "")

#         # Body
#         body = extract_email_body(payload)

#         emails.append({
#             "subject": subject,
#             "body": body,
#             "from": sender,
#             "date": internal_date,
#             "message_id": msg["id"],
#             "text": f"""
# Subject: {subject}

# From: {sender}

# Body:
# {body}
# """
#         })

#     return emails

def fetch_emails(max_results=10):
    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        maxResults=max_results
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"]
        ).execute()

        payload = msg_data.get("payload", {})
        headers = payload.get("headers", [])

        subject = next(
            (h["value"] for h in headers if h["name"] == "Subject"),
            "No Subject"
        )

        sender = next(
            (h["value"] for h in headers if h["name"] == "From"),
            "Unknown"
        )

        attachments = extract_attachments(payload)
        body = extract_email_body(payload)

        attachment_texts = []

        # -------------------------
        # PROCESS ATTACHMENTS
        # -------------------------
        for att in attachments:
            try:
                if att.get("mimeType") == "application/pdf":

                    data = get_attachment(
                        service,
                        msg["id"],
                        att["attachment_id"]
                    )

                    # safe temp file per email (NO overwrite issues)
                    temp_path = f"temp_{msg['id']}.pdf"

                    with open(temp_path, "wb") as f:
                        f.write(data)

                    from services.ingestion import extract_text_from_pdf

                    pdf_text = extract_text_from_pdf(temp_path)
                    attachment_texts.append(pdf_text)

                    # optional cleanup
                    os.remove(temp_path)

            except Exception as e:
                print(f"[ATTACHMENT ERROR] {msg['id']} -> {e}")

        # -------------------------
        # MERGE EVERYTHING
        # -------------------------
        full_text = body + "\n\n" + "\n\n".join(attachment_texts)

        emails.append({
            "message_id": msg["id"],
            "subject": subject,
            "sender": sender,
            "body": body,
            "attachments": attachments,

            # IMPORTANT: THIS is what your RAG uses
            "text": f"""
Subject: {subject}
From: {sender}

{full_text}
""".strip()
        })

    return emails