"""
Email Service for IRIS
Handles sending email notifications about new papers and updates.
"""

import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

import smtplib


logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending email notifications."""

    def __init__(
        self,
        sender: str,
        smtp_server: str,
        smtp_port: int,
        password: str,
        receiver: str,
        subject_prefix: str = "[IRIS] "
    ):
        """
        Initialize email service.

        Args:
            sender: Sender email address
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            password: Email password or app-specific password
            receiver: Receiver email address
            subject_prefix: Prefix for email subjects
        """
        self.sender = sender
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.password = password
        self.receiver = receiver
        self.subject_prefix = subject_prefix

        logger.info(f"Email service initialized: {sender} -> {receiver}")

    def send_notification(
        self,
        subject: str,
        content: str,
        html_content: Optional[str] = None
    ) -> bool:
        """
        Send an email notification.

        Args:
            subject: Email subject
            content: Plain text email content
            html_content: Optional HTML email content

        Returns:
            True if successful, False otherwise
        """
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = self.subject_prefix + subject
        message["From"] = self.sender
        message["To"] = self.receiver

        # Add plain text part
        text_part = MIMEText(content, "plain", "utf-8")
        message.attach(text_part)

        # Add HTML part if provided
        if html_content:
            html_part = MIMEText(html_content, "html", "utf-8")
            message.attach(html_part)

        try:
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.receiver, message.as_string())

            logger.info(f"Email sent successfully: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_update_notification(
        self,
        update_folder: Path,
        new_papers_count: int,
        summaries: Optional[str] = None,
        knowledge_log: Optional[str] = None
    ) -> bool:
        """
        Send notification about IRIS update.

        Args:
            update_folder: Path to the update folder
            new_papers_count: Number of new papers in this update
            summaries: Optional paper summaries
            knowledge_log: Optional knowledge extraction log

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime

        # Create subject
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        subject = f"Update Complete - {new_papers_count} New Papers - {timestamp}"

        # Create email content
        content = f"""
IRIS Update Notification
=====================

Update Time: {timestamp}
New Papers: {new_papers_count}
Update Folder: {update_folder}

This update includes {new_papers_count} new papers from arXiv.
The papers have been processed and added to the knowledge base.

"""

        # Add summaries if available
        if summaries:
            content += "\n" + "=" * 50 + "\n"
            content += "PAPER SUMMARIES\n"
            content += "=" * 50 + "\n\n"
            content += summaries[:5000]  # Limit length
            if len(summaries) > 5000:
                content += "\n\n... (truncated)"
            content += "\n"

        # Add knowledge log if available
        if knowledge_log:
            content += "\n" + "=" * 50 + "\n"
            content += "KNOWLEDGE EXTRACTION\n"
            content += "=" * 50 + "\n\n"
            content += knowledge_log[:3000]  # Limit length
            if len(knowledge_log) > 3000:
                content += "\n\n... (truncated)"
            content += "\n"

        content += "\n" + "-" * 50 + "\n"
        content += "To query the knowledge base, use:\n"
        content += "  python scripts/iris_query.py \"your question\""
        content += "\n"

        # Create HTML version
        html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .section {{ margin-top: 20px; }}
        .stats {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>IRIS Update Complete</h2>
    </div>
    <div class="content">
        <div class="stats">
            <p><strong>Update Time:</strong> {timestamp}</p>
            <p><strong>New Papers:</strong> {new_papers_count}</p>
            <p><strong>Update Folder:</strong> {update_folder}</p>
        </div>

        <div class="section">
            <h3>Paper Summaries</h3>
            <p>{new_papers_count} new papers have been processed and added to the knowledge base.</p>
        </div>

        <div class="section">
            <h3>Query the Knowledge Base</h3>
            <p>To ask questions about the new papers, use:</p>
            <pre><code>python scripts/iris_query.py "your question"</code></pre>
        </div>
    </div>
</body>
</html>
"""

        return self.send_notification(subject, content, html_content)
