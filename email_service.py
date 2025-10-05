#!/usr/bin/env python3
"""
Email Alert Service using Gmail API
Provides air quality alerts via email notifications
"""

import json
import base64
import datetime
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Credentials files
CREDENTIALS_FILE = 'gmail_credentials.json'
TOKEN_FILE = 'gmail_token.json'

class EmailAlertService:
    """Email Alert service for air quality notifications"""
    
    def __init__(self):
        """Initialize Gmail API client"""
        try:
            self.service = self._authenticate_gmail()
            self.subscribers = {}  # In production, use a database
            self.sender_email = None
            
            if self.service:
                # Get sender email address
                try:
                    profile = self.service.users().getProfile(userId='me').execute()
                    self.sender_email = profile.get('emailAddress')
                    print(f"✅ Email Alert Service initialized")
                    print(f"📧 Sender email: {self.sender_email}")
                except Exception as e:
                    print(f"⚠️ Could not get sender email: {e}")
                    self.sender_email = "NASA Air Quality Alerts"
            else:
                print("❌ Email Alert Service initialization failed")
                
        except Exception as e:
            print(f"❌ Email Service initialization failed: {e}")
            self.service = None
            self.sender_email = None
    
    def _authenticate_gmail(self):
        """Authenticate and return Gmail service"""
        creds = None
        
        # Load existing token
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"⚠️ Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(CREDENTIALS_FILE):
                    print("❌ Gmail credentials file not found!")
                    print("💡 Please download credentials from Google Cloud Console")
                    return None
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"❌ OAuth flow failed: {e}")
                    return None
            
            # Save the credentials for the next run
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        try:
            service = build('gmail', 'v1', credentials=creds)
            return service
        except Exception as e:
            print(f"❌ Failed to build Gmail service: {e}")
            return None
    
    def subscribe_user(self, email: str, location: str = "", alert_threshold: int = 100) -> Dict:
        """
        Subscribe a user to email alerts
        
        Args:
            email: User's email address
            location: User's preferred location for alerts
            alert_threshold: AQI threshold for alerts (default: 100)
        
        Returns:
            Dict with subscription status
        """
        try:
            if not self.service:
                return {"success": False, "error": "Email service not available"}
            
            # Validate email format
            if '@' not in email or '.' not in email:
                return {"success": False, "error": "Invalid email format"}
            
            # Store subscription
            self.subscribers[email] = {
                'location': location,
                'alert_threshold': alert_threshold,
                'subscribed_at': datetime.datetime.now().isoformat(),
                'active': True
            }
            
            # Send welcome email
            welcome_subject = "🌍 Welcome to NASA Air Quality Alerts!"
            welcome_html = self._create_welcome_email(email, location, alert_threshold)
            
            if self.send_email(email, welcome_subject, welcome_html):
                return {
                    "success": True,
                    "message": "Successfully subscribed to email alerts",
                    "email": email,
                    "threshold": alert_threshold
                }
            else:
                return {"success": False, "error": "Failed to send welcome email"}
            
        except Exception as e:
            return {"success": False, "error": f"Subscription failed: {str(e)}"}
    
    def unsubscribe_user(self, email: str) -> Dict:
        """Unsubscribe a user from email alerts"""
        try:
            if email in self.subscribers:
                self.subscribers[email]['active'] = False
                
                # Send goodbye email
                goodbye_subject = "👋 Unsubscribed from NASA Air Quality Alerts"
                goodbye_html = self._create_goodbye_email(email)
                
                self.send_email(email, goodbye_subject, goodbye_html)
                
                return {"success": True, "message": "Successfully unsubscribed"}
            else:
                return {"success": False, "error": "Email not found in subscribers"}
                
        except Exception as e:
            return {"success": False, "error": f"Unsubscribe failed: {str(e)}"}
    
    def send_test_email(self, email: str) -> Dict:
        """Send a test email"""
        try:
            if not self.service:
                return {"success": False, "error": "Email service not available"}
            
            subject = "🧪 NASA Air Quality Test Alert"
            html_content = self._create_test_email()
            
            if self.send_email(email, subject, html_content):
                return {"success": True, "message": "Test email sent successfully"}
            else:
                return {"success": False, "error": "Failed to send test email"}
                
        except Exception as e:
            return {"success": False, "error": f"Test email failed: {str(e)}"}
    
    def send_email(self, to_email: str, subject: str, html_content: str) -> bool:
        """
        Send email via Gmail API
        
        Args:
            to_email: Recipient's email address
            subject: Email subject
            html_content: HTML email content
            
        Returns:
            bool: Success status
        """
        if not self.service:
            print("❌ Gmail service not initialized")
            return False
        
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['to'] = to_email
            message['from'] = self.sender_email or 'NASA Air Quality Alerts'
            message['subject'] = subject
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            message.attach(html_part)
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send message
            send_result = self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            print(f"✅ Email sent to {to_email}: {send_result.get('id')}")
            return True
            
        except HttpError as error:
            print(f"❌ Gmail API error: {error}")
            return False
        except Exception as e:
            print(f"❌ Email send failed to {to_email}: {e}")
            return False
    
    def send_aqi_alert(self, email: str, aqi_data: Dict) -> bool:
        """Send AQI alert email"""
        try:
            aqi_value = aqi_data.get('overall_aqi', 0)
            location = aqi_data.get('location', {}).get('address', 'Unknown Location')
            
            subject = f"🚨 Air Quality Alert: {location} - AQI {aqi_value}"
            html_content = self._create_aqi_alert_email(aqi_data)
            
            return self.send_email(email, subject, html_content)
            
        except Exception as e:
            print(f"❌ Failed to send AQI alert: {e}")
            return False
    
    def _create_welcome_email(self, email: str, location: str, threshold: int) -> str:
        """Create welcome email HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .footer {{ background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
                .alert-levels {{ margin: 20px 0; }}
                .level {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .good {{ background: #d4edda; color: #155724; }}
                .moderate {{ background: #fff3cd; color: #856404; }}
                .unhealthy {{ background: #f8d7da; color: #721c24; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌍 Welcome to NASA Air Quality Alerts!</h1>
                    <p>Stay informed about air quality in your area</p>
                </div>
                <div class="content">
                    <h2>🎉 Subscription Confirmed</h2>
                    <p><strong>Email:</strong> {email}</p>
                    <p><strong>Location:</strong> {location or 'Auto-detect'}</p>
                    <p><strong>Alert Threshold:</strong> {threshold} AQI</p>
                    
                    <h3>📊 What to Expect</h3>
                    <div class="alert-levels">
                        <div class="level good">🟢 Good (0-50): No alerts</div>
                        <div class="level moderate">🟡 Moderate (51-100): Sensitive groups may receive alerts</div>
                        <div class="level unhealthy">🔴 Unhealthy (101+): Everyone receives alerts</div>
                    </div>
                    
                    <h3>🛡️ Health Benefits</h3>
                    <ul>
                        <li>Real-time air quality monitoring</li>
                        <li>Health recommendations based on current conditions</li>
                        <li>NASA-powered ML predictions</li>
                        <li>Personalized alerts for your location</li>
                    </ul>
                </div>
                <div class="footer">
                    <p>NASA Space Apps Challenge 2024 | Air Quality Monitoring System</p>
                    <p><small>Reply "UNSUBSCRIBE" to stop receiving alerts</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_goodbye_email(self, email: str) -> str:
        """Create goodbye email HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #6c757d; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .footer {{ background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>👋 Goodbye from NASA Air Quality Alerts</h1>
                </div>
                <div class="content">
                    <h2>Unsubscribed Successfully</h2>
                    <p>You have been unsubscribed from air quality alerts for <strong>{email}</strong>.</p>
                    <p>Thank you for using our NASA-powered air quality monitoring system!</p>
                    <p>You can resubscribe anytime at our web interface.</p>
                    
                    <h3>🌟 We Hope You Enjoyed</h3>
                    <ul>
                        <li>Real-time air quality data</li>
                        <li>NASA machine learning predictions</li>
                        <li>Personalized health recommendations</li>
                    </ul>
                </div>
                <div class="footer">
                    <p>NASA Space Apps Challenge 2024 | Air Quality Monitoring System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_test_email(self) -> str:
        """Create test email HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
                .content { background: #f8f9fa; padding: 30px; }
                .footer { background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧪 Test Email Successful!</h1>
                </div>
                <div class="content">
                    <h2>✅ Email System Working</h2>
                    <p>This is a test email from the NASA Air Quality Alert System.</p>
                    <p><strong>Timestamp:</strong> """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                    
                    <h3>🚀 System Status</h3>
                    <ul>
                        <li>✅ Gmail API Connection: Active</li>
                        <li>✅ Email Delivery: Working</li>
                        <li>✅ HTML Formatting: Enabled</li>
                        <li>✅ NASA Integration: Ready</li>
                    </ul>
                </div>
                <div class="footer">
                    <p>NASA Space Apps Challenge 2024 | Air Quality Monitoring System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _create_aqi_alert_email(self, aqi_data: Dict) -> str:
        """Create AQI alert email HTML"""
        aqi_value = aqi_data.get('overall_aqi', 0)
        category = aqi_data.get('overall_category', 'Unknown')
        location = aqi_data.get('location', {}).get('address', 'Unknown Location')
        
        # Get color based on AQI
        if aqi_value <= 50:
            color = "#28a745"
            emoji = "🟢"
        elif aqi_value <= 100:
            color = "#ffc107"
            emoji = "🟡"
        elif aqi_value <= 150:
            color = "#fd7e14"
            emoji = "🟠"
        elif aqi_value <= 200:
            color = "#dc3545"
            emoji = "🔴"
        elif aqi_value <= 300:
            color = "#6f42c1"
            emoji = "🟣"
        else:
            color = "#6c757d"
            emoji = "🚨"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: {color}; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .footer {{ background: #343a40; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
                .aqi-box {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; border: 3px solid {color}; }}
                .aqi-value {{ font-size: 3em; font-weight: bold; color: {color}; }}
                .recommendations {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{emoji} Air Quality Alert</h1>
                    <h2>📍 {location}</h2>
                </div>
                <div class="content">
                    <div class="aqi-box">
                        <div class="aqi-value">{aqi_value}</div>
                        <h3>{category}</h3>
                        <p><strong>Current Air Quality Index</strong></p>
                    </div>
                    
                    <h3>🏥 Health Recommendations</h3>
                    <div class="recommendations">
                        {self._get_health_advice_html(aqi_value)}
                    </div>
                    
                    <h3>📊 Additional Information</h3>
                    <ul>
                        <li><strong>Timestamp:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
                        <li><strong>Data Source:</strong> NASA ML Models + OpenAQ</li>
                        <li><strong>Alert Threshold:</strong> Your configured level</li>
                    </ul>
                </div>
                <div class="footer">
                    <p>NASA Space Apps Challenge 2024 | Air Quality Monitoring System</p>
                    <p><small>Reply "UNSUBSCRIBE" to stop receiving alerts</small></p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_health_advice_html(self, aqi: int) -> str:
        """Get health advice HTML based on AQI level"""
        if aqi <= 50:
            return "✅ <strong>Good air quality.</strong> Enjoy outdoor activities!"
        elif aqi <= 100:
            return "⚠️ <strong>Moderate air quality.</strong> Sensitive people should limit prolonged outdoor activities."
        elif aqi <= 150:
            return "🚫 <strong>Unhealthy for sensitive groups.</strong> Children, elderly, and people with respiratory conditions should limit outdoor activities."
        elif aqi <= 200:
            return "⛔ <strong>Unhealthy air quality.</strong> Everyone should avoid prolonged outdoor activities."
        elif aqi <= 300:
            return "🚨 <strong>Very unhealthy!</strong> Everyone should avoid outdoor activities and stay indoors."
        else:
            return "🆘 <strong>HAZARDOUS!</strong> Emergency conditions. Stay indoors and seek medical attention if experiencing symptoms."
    
    def get_service_status(self) -> Dict:
        """Get email service status"""
        active_subscribers = sum(1 for sub in self.subscribers.values() if sub.get('active', True))
        
        return {
            "service_active": self.service is not None,
            "sender_email": self.sender_email,
            "credentials_configured": os.path.exists(CREDENTIALS_FILE),
            "token_available": os.path.exists(TOKEN_FILE),
            "total_subscribers": len(self.subscribers),
            "active_subscribers": active_subscribers,
            "messages_sent_today": 0  # Would implement with proper logging
        }
    
    def get_subscriber_count(self) -> int:
        """Get total number of active subscribers"""
        return sum(1 for sub in self.subscribers.values() if sub.get('active', False))
    
    def get_subscriber_info(self, email: str) -> Optional[Dict]:
        """Get subscriber information"""
        return self.subscribers.get(email)
    
    def check_and_send_alerts(self, aqi_data: Dict) -> List[str]:
        """Check all subscribers and send alerts if needed"""
        alerts_sent = []
        overall_aqi = aqi_data.get('overall_aqi', 0)
        
        for email, subscription in self.subscribers.items():
            if not subscription.get('active', False):
                continue
                
            threshold = subscription.get('alert_threshold', 100)
            
            if overall_aqi >= threshold:
                if self.send_aqi_alert(email, aqi_data):
                    alerts_sent.append(email)
        
        return alerts_sent


# Global email service instance
email_service = EmailAlertService()