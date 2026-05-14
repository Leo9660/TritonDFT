import os
import resend

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "TritonDFT <admin@tritondft.com>")
LOGO_URL = os.environ.get("LOGO_URL", "https://chat.tritondft.com/logo.png")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY


def _magic_link_html(link: str) -> str:
    return f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Sign in to TritonDFT</title>
</head>
<body style="margin:0;padding:0;background-color:#f4f5f7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased;">

<div style="display:none;font-size:1px;color:#f4f5f7;line-height:1px;max-height:0px;max-width:0px;opacity:0;overflow:hidden;">Your one-time sign-in link to TritonDFT. Expires in 15 minutes.</div>

<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color:#f4f5f7;padding:40px 16px;">
  <tr>
    <td align="center">

      <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width:480px;background:#ffffff;border-radius:16px;box-shadow:0 1px 3px rgba(20,30,60,0.04),0 12px 28px rgba(20,30,60,0.08);overflow:hidden;">

        <tr>
          <td style="background:#0d1018;background-image:linear-gradient(135deg,#4577ff 0%,#1e40af 60%,#0b1e6e 100%);padding:36px 24px 28px;text-align:center;">
            <img src="{LOGO_URL}" alt="TritonDFT" width="56" height="56" style="display:inline-block;width:56px;height:56px;border-radius:14px;background:#ffffff;padding:4px;box-shadow:0 4px 12px rgba(0,0,0,0.18);" />
            <div style="margin-top:14px;font-family:Georgia,'Times New Roman',serif;font-style:italic;font-weight:400;font-size:18px;color:#ffffff;letter-spacing:0.2px;">
              TritonDFT
            </div>
          </td>
        </tr>

        <tr>
          <td style="padding:36px 40px 0;text-align:center;">
            <h1 style="margin:0 0 14px;font-family:Georgia,'Times New Roman',serif;font-style:italic;font-weight:400;font-size:26px;line-height:1.25;color:#0f172a;">
              Sign in to your account
            </h1>
            <p style="margin:0 0 28px;font-size:15px;line-height:1.55;color:#475569;">
              Tap the button below to sign in. This link works once and expires in <strong style="color:#0f172a;font-weight:600;">15 minutes</strong>.
            </p>
          </td>
        </tr>

        <tr>
          <td align="center" style="padding:0 40px;">
            <table role="presentation" cellpadding="0" cellspacing="0" border="0">
              <tr>
                <td bgcolor="#2b5bff" style="border-radius:10px;background:#2b5bff;background-image:linear-gradient(135deg,#4577ff 0%,#2b5bff 100%);box-shadow:0 6px 16px rgba(43,91,255,0.32);">
                  <a href="{link}" target="_blank" style="display:inline-block;padding:14px 40px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:15px;font-weight:600;color:#ffffff;text-decoration:none;border-radius:10px;letter-spacing:0.2px;">
                    Sign in &rarr;
                  </a>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <tr>
          <td style="padding:28px 40px 0;text-align:center;">
            <p style="margin:0 0 6px;font-size:12px;color:#94a3b8;letter-spacing:0.1px;">
              Or paste this URL into your browser:
            </p>
            <p style="margin:0;font-size:12px;line-height:1.5;word-break:break-all;">
              <a href="{link}" style="color:#4577ff;text-decoration:none;">{link}</a>
            </p>
          </td>
        </tr>

        <tr>
          <td style="padding:28px 40px 32px;">
            <div style="border-top:1px solid #e2e8f0;padding-top:18px;">
              <p style="margin:0;font-size:12px;line-height:1.55;color:#64748b;">
                <span style="color:#0f172a;font-weight:600;">Didn&rsquo;t request this?</span><br/>
                You can safely ignore this email &mdash; nobody can sign in without clicking the button above. We&rsquo;ll never ask you for your password.
              </p>
            </div>
          </td>
        </tr>
      </table>

      <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width:480px;margin-top:24px;">
        <tr>
          <td align="center" style="padding:0 16px;">
            <p style="margin:0;font-size:11px;line-height:1.6;color:#94a3b8;">
              <a href="https://tritondft.com" style="color:#94a3b8;text-decoration:none;font-weight:600;">TritonDFT</a> &middot; LLM-driven density functional theory<br/>
              Built at <a href="https://yufeiding.ucsd.edu/" style="color:#94a3b8;text-decoration:underline;">Picasso Lab</a>, UC San Diego &middot; La Jolla, CA
            </p>
          </td>
        </tr>
      </table>

    </td>
  </tr>
</table>

</body>
</html>"""


def _magic_link_text(link: str) -> str:
    return (
        "Sign in to TritonDFT\n"
        "========================\n\n"
        "Tap the link below to sign in. It works once and expires in 15 minutes:\n\n"
        f"{link}\n\n"
        "Didn't request this? You can safely ignore this email.\n\n"
        "--\n"
        "TritonDFT - LLM-driven density functional theory\n"
        "Built at Picasso Lab, UC San Diego\n"
    )


def send_magic_link_email(to: str, link: str):
    """Send a one-time login link via Resend (HTML + plain-text)."""
    if not RESEND_API_KEY:
        return None, "RESEND_API_KEY not configured"
    try:
        result = resend.Emails.send({
            "from": EMAIL_FROM,
            "to": [to],
            "subject": "Your TritonDFT sign-in link",
            "html": _magic_link_html(link),
            "text": _magic_link_text(link),
        })
        rid = result.get("id") if isinstance(result, dict) else None
        return rid, None
    except Exception as e:
        return None, str(e)
