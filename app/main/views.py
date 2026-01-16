from flask import (
    url_for,
    render_template,
    redirect,
    url_for,
    abort,
    flash,
    request,
    current_app,
    jsonify,
    session,
    send_from_directory,
)
from . import main
from flask_login import current_user
from app.utils.decorators import *
from app.utils.misc import generate_layout
from app.utils.authorizer import Authorizer
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os

import json
# Scopes (keep read-only for evidence collection)
GOOGLE_SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.metadata.readonly'
]



@main.route("/test", methods=["GET"])
@login_required
def test():
    if request.args.get("2"):
        return render_template("policy_center.html")
    return render_template("test.html")


@main.route("/assessments/<string:id>/manage", methods=["GET"])
@login_required
def get_assessment_for_edit_mode(id):
    result = Authorizer(current_user).can_user_manage_assessment(id)
    return render_template("kanban.html", assessment=result["extra"]["assessment"])


@main.route("/assessments/<string:id>", methods=["GET"])
@login_required
def view_assessment(id):
    """
    Endpoint is for vendors to load and respond to assessments
    See view_assessment_overview for the endpoint for InfoSec
    to review assessments
    """
    result = Authorizer(current_user).can_user_respond_to_assessment(id)
    if not result["extra"]["assessment"].is_assessment_published():
        flash("Assessment is not published", "warning")
        return redirect(url_for("main.home"))
    return render_template("kanban_view.html", assessment=result["extra"]["assessment"])


@main.route("/assessments/<string:id>/review", methods=["GET"])
@login_required
def view_assessment_overview(id):
    """
    Endpoint is for InfoSec to review assessments
    See view_assessment for the endpoint that vendors use
    to respond to assessments
    """
    result = Authorizer(current_user).can_user_manage_assessment(id)
    return render_template(
        "assessment_overview.html", assessment=result["extra"]["assessment"]
    )


@main.route("/forms/<string:id>", methods=["GET"])
@login_required
def view_form(id):
    result = Authorizer(current_user).can_user_read_form(id)
    return render_template("view_form.html", form=result["extra"]["form"])


@main.route("/", methods=["GET"])
@login_required
def home():
    return render_template("home.html")


@main.route("/projects/<string:pid>/reports/<path:filename>", methods=["GET"])
@login_required
def download_report(pid, filename):
    result = Authorizer(current_user).can_user_access_project(pid)
    return send_from_directory(
        directory=current_app.config["UPLOAD_FOLDER"], path=filename, as_attachment=True
    )


@main.route("/frameworks", methods=["GET"])
@login_required
def frameworks():
    return render_template("frameworks.html")


@main.route("/tenants/<string:id>/risk", methods=["GET"])
@login_required
def risks(id):
    Authorizer(current_user).can_user_access_risk_module(id)
    return render_template("risk_register.html")


@main.route("/assessments", methods=["GET"])
@login_required
def assessments():
    return render_template("assessments.html")


@main.route("/policies", methods=["GET"])
@login_required
def policies():
    return render_template("policies.html")


@main.route("/tenants/<string:id>/policy-center", methods=["GET"])
@login_required
def view_policy_center(id):
    Authorizer(current_user).can_user_access_tenant(id)
    policy_id = request.args.get("policy-id")
    return render_template("pc.html", tenant_id=id, policy_id=policy_id)


@main.route("/projects", methods=["GET"])
@login_required
def projects():
    return render_template("projects.html")


@main.route("/projects/<string:pid>", methods=["GET"])
@login_required
def view_project(pid):
    result = Authorizer(current_user).can_user_access_project(pid)
    return render_template("view_project.html", project=result["extra"]["project"])


@main.route("/projects/<string:pid>/controls/<string:cid>", methods=["GET"])
@login_required
def view_control_in_project(pid, cid):
    result = Authorizer(current_user).can_user_read_project_control(cid)
    return render_template(
        "view_control_in_project.html",
        project=result["extra"]["control"].project,
        project_control=result["extra"]["control"],
    )


@main.route("/projects/<string:id>/policy-center", methods=["GET"])
@login_required
def view_policy_center_for_project(id):
    result = Authorizer(current_user).can_user_read_project(id)
    policy_id = request.args.get("policy-id")
    return render_template(
        "policy_center.html", project=result["extra"]["project"], policy_id=policy_id
    )


@main.route("/labels", methods=["GET"])
@login_required
def labels():
    return render_template("labels.html")


@main.route("/tags", methods=["GET"])
@login_required
def tags():
    return render_template("tags.html")


@main.route("/vendors/<string:id>", methods=["GET"])
@login_required
def get_vendor(id):
    result = Authorizer(current_user).can_user_access_vendor(id)
    vendor = result["extra"]["vendor"]
    return render_template("view_vendor.html", vendor=vendor)


@main.route("/applications/<string:id>", methods=["GET"])
@login_required
def get_application(id):
    result = Authorizer(current_user).can_user_access_application(id)
    application = result["extra"]["application"]
    return render_template("view_application.html", application=application)


@main.route("/search-vendors", methods=["GET"])
@login_required
def search_vendor():
    # TODO - auth
    return render_template("search_vendor.html")

@main.route('/integrations')
@login_required
def integrations():
    return render_template('integrations.html')

@main.route('/ai')
@login_required
def ai():
    return render_template('ai.html')

@main.route('/integrations/google/connect')
@login_required
def google_connect():
    """Start Google OAuth flow - only logged-in users can initiate"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": current_app.config['GOOGLE_CLIENT_ID'],
                "client_secret": current_app.config['GOOGLE_CLIENT_SECRET'],
                "redirect_uris": [current_app.config['GOOGLE_REDIRECT_URI']],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=GOOGLE_SCOPES
    )
    flow.redirect_uri = current_app.config['GOOGLE_REDIRECT_URI']

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )

    session['google_oauth_state'] = state
    session.modified = True  # Force session save

    return redirect(authorization_url)


@main.route('/integrations/google/callback')
def google_callback():
    """Handle Google OAuth callback - NO @login_required here to avoid loop"""
    state = session.pop('google_oauth_state', None)

    if not state:
        flash("Invalid or expired OAuth state – please try connecting again", "error")
        return redirect(url_for('main.integrations'))

    # Re-create Flow object (safe & lightweight)
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": current_app.config['GOOGLE_CLIENT_ID'],
                "client_secret": current_app.config['GOOGLE_CLIENT_SECRET'],
                "redirect_uris": [current_app.config['GOOGLE_REDIRECT_URI']],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=GOOGLE_SCOPES,
        redirect_uri=current_app.config['GOOGLE_REDIRECT_URI']
    )

    try:
        flow.fetch_token(
            authorization_response=request.url,
            state=state
        )
        credentials = flow.credentials
        session['google_credentials'] = credentials.to_json()
        session.modified = True  # Ensure session is saved

        flash("Successfully connected to Google Workspace!", "success")
    except Exception as e:
        current_app.logger.error(f"Google callback error: {str(e)}")
        flash(f"Connection failed: {str(e)}", "error")

    return redirect(url_for('main.integrations'))


@main.route('/integrations/google/files')
@login_required
def google_files():
    """List recent files from the connected user's Google Drive"""
    creds_json = session.get('google_credentials')
    if not creds_json:
        flash("Not connected to Google Drive", "warning")
        return redirect(url_for('main.integrations'))

    try:
        creds = Credentials.from_authorized_user_info(json.loads(creds_json))
        service = build('drive', 'v3', credentials=creds)

        results = service.files().list(
            pageSize=10,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)"
        ).execute()
        files = results.get('files', [])
        return jsonify({"files": files})
    except Exception as e:
        flash(f"Error fetching files: {str(e)}", "error")
        return jsonify({"error": str(e)}), 500


@main.route('/integrations/google/me')
@login_required
def google_me():
    """Get basic user info from connected Google account (for frontend display)"""
    creds_json = session.get('google_credentials')
    if not creds_json:
        return jsonify({"error": "Not connected"}), 401

    try:
        creds = Credentials.from_authorized_user_info(json.loads(creds_json))
        service = build('oauth2', 'v2', credentials=creds)
        user_info = service.userinfo().get().execute()
        return jsonify({
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture"),
            "connected": True
        })
    except Exception as e:
        current_app.logger.error(f"Google me error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@main.route('/integrations/google/disconnect', methods=['POST'])
@login_required
def google_disconnect():
    """Disconnect user's Google account"""
    session.pop('google_credentials', None)
    session.pop('google_oauth_state', None)
    session.modified = True
    flash("Google Workspace disconnected", "info")
    return jsonify({"status": "disconnected"})


@main.route('/integrations/google/sheets')
@login_required
def google_sheets():
    """List Google Sheets (useful for Google Forms responses)"""
    creds_json = session.get('google_credentials')
    if not creds_json:
        flash("Connect Google first", "warning")
        return redirect(url_for('main.integrations'))

    try:
        creds = Credentials.from_authorized_user_info(json.loads(creds_json))
        service = build('drive', 'v3', credentials=creds)

        results = service.files().list(
            q="mimeType='application/vnd.google-apps.spreadsheet'",
            pageSize=15,
            fields="files(id, name, modifiedTime, webViewLink, owners(emailAddress))"
        ).execute()
        sheets = results.get('files', [])

        return render_template('google_sheets.html', sheets=sheets)
    except Exception as e:
        flash(f"Error loading sheets: {str(e)}", "error")
        return redirect(url_for('main.integrations'))