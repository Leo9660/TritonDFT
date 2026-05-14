import os
import resend

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "TritonDFT <admin@picasso-lab.com>")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY


def send_magic_link_email(to: str, link: str):
    """Send a one-time login link via Resend."""
    if not RESEND_API_KEY:
        return None, "RESEND_API_KEY not configured"
    html = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:480px;margin:0 auto;padding:24px;color:#222;">
            <h2 style="margin:0 0 16px;">Sign in to TritonDFT</h2>
            <p style="color:#555;line-height:1.5;">Click below to log in. This link expires in 15 minutes and can only be used once.</p>
            <p style="margin:24px 0;">
                <a href="{link}" style="background:#3b5bff;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;display:inline-block;">Sign in</a>
            </p>
            <p style="color:#888;font-size:12px;">If the button doesn't work, paste this URL into your browser:<br/><span style="word-break:break-all;">{link}</span></p>
            <p style="color:#888;font-size:12px;margin-top:24px;">If you didn't request this, you can safely ignore this email.</p>
        </div>
    """
    try:
        result = resend.Emails.send({
            "from": EMAIL_FROM,
            "to": [to],
            "subject": "Your TritonDFT login link",
            "html": html,
        })
        rid = result.get("id") if isinstance(result, dict) else None
        return rid, None
    except Exception as e:
        return None, str(e)
