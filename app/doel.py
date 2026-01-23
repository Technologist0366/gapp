# app/doel.py - Global Evidence Repository Management
import os
import json
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func
from flask import current_app, abort
from werkzeug.datastructures import FileStorage

class GlobalEvidenceRepository:
    """Manages global evidence repository operations"""
    
    @staticmethod
    def get_all_evidence(tenant_id, user, filters=None, page=1, per_page=50):
        """
        Get all evidence across all projects for a tenant
        
        Args:
            tenant_id: Tenant ID
            user: Current user object
            filters: Dictionary of filters
            page: Page number
            per_page: Items per page
        
        Returns:
            Paginated evidence items
        """
        from app.models import ProjectEvidence, Project
        from app.utils.authorizer import Authorizer
        
        # Get query using the class method
        query = ProjectEvidence.get_global_evidence_for_tenant(
            tenant_id=tenant_id,
            filters=filters
        )
        
        # Paginate
        paginated = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # Format results
        evidence_items = []
        for evidence in paginated.items:
            evidence_data = evidence.to_global_dict()
            evidence_items.append(evidence_data)
        
        return {
            'evidence': evidence_items,
            'pagination': {
                'page': paginated.page,
                'per_page': paginated.per_page,
                'total': paginated.total,
                'pages': paginated.pages
            }
        }
    
    @staticmethod
    def get_statistics(tenant_id):
        """
        Get statistics for global evidence repository
        
        Args:
            tenant_id: Tenant ID
        
        Returns:
            Dictionary of statistics
        """
        from app.models import ProjectEvidence, Project
        
        stats = {
            'total': 0,
            'ai_extracted': 0,
            'manual': 0,
            'uploaded': 0,
            'needs_review': 0,
            'this_week': 0,
            'high_severity': 0,
            'global': 0,
            'linked': 0,
            'projects': 0,
            'linked_risks': 0  # Placeholder for risk integration
        }
        
        # Base query
        query = ProjectEvidence.query.filter(
            ProjectEvidence.tenant_id == tenant_id
        )
        
        # Total evidence
        stats['total'] = query.count()
        
        # Count by source type
        stats['ai_extracted'] = query.filter(
            ProjectEvidence.source_type == 'ai_ocr'
        ).count()
        
        stats['manual'] = query.filter(
            or_(
                ProjectEvidence.source_type == 'manual',
                ProjectEvidence.source_type.is_(None)
            )
        ).count()
        
        stats['uploaded'] = query.filter(
            ProjectEvidence.source_type == 'upload'
        ).count()
        
        # Needs review
        stats['needs_review'] = query.filter(
            ProjectEvidence.needs_review == True
        ).count()
        
        # This week's evidence
        week_ago = datetime.utcnow() - timedelta(days=7)
        stats['this_week'] = query.filter(
            ProjectEvidence.date_added >= week_ago
        ).count()
        
        # High severity
        stats['high_severity'] = query.filter(
            ProjectEvidence.severity.in_(['high', 'critical'])
        ).count()
        
        # Global vs linked
        stats['global'] = query.filter(
            ProjectEvidence.project_id.is_(None)
        ).count()
        
        stats['linked'] = query.filter(
            ProjectEvidence.project_id.isnot(None)
        ).count()
        
        # Number of projects
        stats['projects'] = Project.query.filter_by(
            tenant_id=tenant_id
        ).count()
        
        return stats
    
    @staticmethod
    def create_global_evidence(tenant_id, owner_id, data, file=None):
        """
        Create global evidence (not linked to any project)
        
        Args:
            tenant_id: Tenant ID
            owner_id: Owner user ID
            data: Evidence data dictionary
            file: Optional file object
        
        Returns:
            Created evidence object
        """
        from app.models import ProjectEvidence, Tenant
        from app import db
        
        tenant = Tenant.query.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")
        
        # Validate required fields
        if not data.get('name'):
            raise ValueError("Evidence name is required")
        
        # Create evidence with no project_id to make it global
        evidence = ProjectEvidence(
            name=data.get('name'),
            description=data.get('description', ''),
            content=data.get('content', ''),
            source_type=data.get('source_type', 'manual'),
            evidence_type=data.get('evidence_type'),
            compliance_standard=data.get('compliance_standard'),
            severity=data.get('severity'),
            needs_review=data.get('needs_review', False),
            extraction_metadata=data.get('extraction_metadata'),
            source_document=data.get('source_document'),
            source_reference=data.get('source_reference'),
            is_global=True,
            owner_id=owner_id,
            tenant_id=tenant_id,
            project_id=None  # No project makes it global
        )
        
        db.session.add(evidence)
        db.session.commit()
        
        # Save file if provided
        if file and isinstance(file, FileStorage):
            try:
                evidence.save_file(
                    file_object=file,
                    file_name=file.filename
                )
            except Exception as e:
                current_app.logger.warning(f"Failed to save file for evidence: {str(e)}")
        
        return evidence
    
    @staticmethod
    def link_to_project(evidence_id, project_id, user_id):
        """
        Link global evidence to a project
        
        Args:
            evidence_id: Evidence ID
            project_id: Project ID to link to
            user_id: User ID performing the action
        
        Returns:
            Updated evidence object
        """
        from app.models import ProjectEvidence, Project
        from app import db
        
        evidence = ProjectEvidence.query.get(evidence_id)
        if not evidence:
            raise ValueError(f"Evidence not found: {evidence_id}")
        
        project = Project.query.get(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        # Check if already linked to this project
        if evidence.project_id == project_id:
            return evidence
        
        # Link evidence to project
        evidence.project_id = project_id
        evidence.tenant_id = project.tenant_id  # Ensure consistent tenant
        evidence.is_global = False  # No longer global
        
        db.session.commit()
        
        # Log the action if tenant has logging
        if hasattr(project.tenant, 'add_log'):
            try:
                project.tenant.add_log(
                    message=f"Linked evidence '{evidence.name}' to project '{project.name}'",
                    namespace="evidence",
                    action="link_to_project",
                    user_id=user_id,
                    meta={
                        'evidence_id': evidence_id,
                        'project_id': project_id
                    }
                )
            except:
                pass  # Logging optional
        
        return evidence
    
    @staticmethod
    def unlink_from_project(evidence_id, user_id):
        """
        Unlink evidence from project (make it global again)
        
        Args:
            evidence_id: Evidence ID
            user_id: User ID performing the action
        
        Returns:
            Updated evidence object
        """
        from app.models import ProjectEvidence
        from app import db
        
        evidence = ProjectEvidence.query.get(evidence_id)
        if not evidence:
            raise ValueError(f"Evidence not found: {evidence_id}")
        
        # Store project info for logging
        project_name = evidence.project.name if evidence.project else None
        
        # Make evidence global
        evidence.project_id = None
        evidence.is_global = True
        
        db.session.commit()
        
        # Log the action
        if hasattr(evidence.tenant, 'add_log'):
            try:
                evidence.tenant.add_log(
                    message=f"Unlinked evidence '{evidence.name}' from project '{project_name}'",
                    namespace="evidence",
                    action="unlink_from_project",
                    user_id=user_id,
                    meta={'evidence_id': evidence_id}
                )
            except:
                pass  # Logging optional
        
        return evidence
    
    @staticmethod
    def search_evidence(tenant_id, query_string, limit=50):
        """
        Search evidence across tenant
        
        Args:
            tenant_id: Tenant ID
            query_string: Search query
            limit: Maximum results
        
        Returns:
            List of matching evidence
        """
        from app.models import ProjectEvidence
        
        if not query_string or not query_string.strip():
            return []
        
        search_query = query_string.strip()
        
        search_results = ProjectEvidence.query.filter(
            ProjectEvidence.tenant_id == tenant_id
        ).filter(
            or_(
                ProjectEvidence.name.ilike(f'%{search_query}%'),
                ProjectEvidence.description.ilike(f'%{search_query}%'),
                ProjectEvidence.content.ilike(f'%{search_query}%')
            )
        ).order_by(
            ProjectEvidence.date_added.desc()
        ).limit(limit).all()
        
        results = []
        for evidence in search_results:
            result = evidence.to_global_dict()
            results.append(result)
        
        return results
    
    @staticmethod
    def get_recent_projects_with_evidence(tenant_id, limit=5):
        """
        Get recent projects with evidence counts
        
        Args:
            tenant_id: Tenant ID
            limit: Maximum number of projects
        
        Returns:
            List of recent projects
        """
        from app.models import Project, ProjectEvidence
        
        # Get projects with recent evidence
        projects = Project.query.filter_by(
            tenant_id=tenant_id
        ).order_by(
            Project.date_updated.desc()
        ).limit(limit * 2).all()  # Get more than limit to filter
        
        recent_projects = []
        for project in projects:
            evidence_count = ProjectEvidence.query.filter_by(
                project_id=project.id
            ).count()
            
            if evidence_count > 0:  # Only include projects with evidence
                recent_projects.append({
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'evidence_count': evidence_count,
                    'updated_at': project.date_updated.isoformat() if project.date_updated else None,
                    'created_at': project.date_added.isoformat() if project.date_added else None,
                    'framework': project.framework.name if project.framework else 'Custom'
                })
            
            if len(recent_projects) >= limit:
                break
        
        return recent_projects
    
    @staticmethod
    def bulk_update_metadata(evidence_ids, metadata_updates, user_id):
        """
        Bulk update evidence metadata
        
        Args:
            evidence_ids: List of evidence IDs
            metadata_updates: Dictionary of metadata to update
            user_id: User ID performing the action
        
        Returns:
            Dictionary with results
        """
        from app.models import ProjectEvidence
        from app import db
        
        allowed_fields = [
            'severity', 'needs_review', 'evidence_type',
            'compliance_standard', 'source_type', 'is_global'
        ]
        
        # Filter updates to only allowed fields
        updates = {}
        for field, value in metadata_updates.items():
            if field in allowed_fields:
                # Convert string booleans
                if field in ['needs_review', 'is_global'] and isinstance(value, str):
                    value = value.lower() == 'true'
                updates[field] = value
        
        if not updates:
            return {'updated': 0, 'errors': ['No valid fields to update']}
        
        updated_count = 0
        errors = []
        updated_evidence_names = []
        
        for evidence_id in evidence_ids:
            try:
                evidence = ProjectEvidence.query.get(evidence_id)
                if not evidence:
                    errors.append(f"Evidence not found: {evidence_id}")
                    continue
                
                # Apply updates
                for field, value in updates.items():
                    if hasattr(evidence, field):
                        setattr(evidence, field, value)
                
                updated_count += 1
                updated_evidence_names.append(evidence.name)
                
            except Exception as e:
                errors.append(f"Error updating evidence {evidence_id}: {str(e)}")
        
        if updated_count > 0:
            db.session.commit()
            
            # Log bulk action
            try:
                if evidence and hasattr(evidence.tenant, 'add_log'):
                    evidence.tenant.add_log(
                        message=f"Bulk updated {updated_count} evidence items",
                        namespace="evidence",
                        action="bulk_update",
                        user_id=user_id,
                        meta={
                            'count': updated_count,
                            'updates': updates,
                            'evidence_names': updated_evidence_names[:10]  # Limit names
                        }
                    )
            except:
                pass  # Logging optional
        
        return {
            'updated': updated_count,
            'errors': errors,
            'message': f'Updated {updated_count} evidence items'
        }
    
    @staticmethod
    def bulk_delete_evidence(evidence_ids, user_id):
        """
        Bulk delete evidence items
        
        Args:
            evidence_ids: List of evidence IDs
            user_id: User ID performing the action
        
        Returns:
            Dictionary with results
        """
        from app.models import ProjectEvidence
        from app import db
        
        deleted_count = 0
        errors = []
        deleted_evidence_names = []
        
        for evidence_id in evidence_ids:
            try:
                evidence = ProjectEvidence.query.get(evidence_id)
                if not evidence:
                    errors.append(f"Evidence not found: {evidence_id}")
                    continue
                
                evidence_name = evidence.name
                
                # Delete the evidence
                db.session.delete(evidence)
                
                deleted_count += 1
                deleted_evidence_names.append(evidence_name)
                
            except Exception as e:
                errors.append(f"Error deleting evidence {evidence_id}: {str(e)}")
        
        if deleted_count > 0:
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                errors.append(f"Database error during bulk delete: {str(e)}")
                deleted_count = 0
            
            # Log bulk action
            try:
                if evidence and hasattr(evidence.tenant, 'add_log'):
                    evidence.tenant.add_log(
                        message=f"Bulk deleted {deleted_count} evidence items",
                        namespace="evidence",
                        action="bulk_delete",
                        user_id=user_id,
                        meta={
                            'count': deleted_count,
                            'evidence_names': deleted_evidence_names[:10]
                        }
                    )
            except:
                pass  # Logging optional
        
        return {
            'deleted': deleted_count,
            'errors': errors,
            'message': f'Deleted {deleted_count} evidence items'
        }
    
    @staticmethod
    def get_evidence_details(evidence_id):
        """
        Get detailed evidence information
        
        Args:
            evidence_id: Evidence ID
        
        Returns:
            Detailed evidence dictionary
        """
        from app.models import ProjectEvidence
        
        evidence = ProjectEvidence.query.get(evidence_id)
        if not evidence:
            raise ValueError(f"Evidence not found: {evidence_id}")
        
        details = evidence.to_global_dict()
        
        # Add additional details
        details['file_info'] = {}
        if evidence.file_name:
            details['file_info'] = {
                'name': evidence.file_name,
                'provider': evidence.file_provider,
                'has_file': True
            }
        
        # Add control associations details
        details['control_associations'] = []
        try:
            controls = evidence.get_controls()
            for control in controls:
                details['control_associations'].append({
                    'id': control.id,
                    'name': control.name if hasattr(control, 'name') else str(control.id),
                    'type': 'subcontrol'
                })
        except:
            pass  # Control associations optional
        
        # Add audit trail if available
        details['audit_trail'] = []
        try:
            if hasattr(evidence, 'date_added'):
                details['audit_trail'].append({
                    'action': 'created',
                    'date': evidence.date_added.isoformat() if evidence.date_added else None,
                    'user_id': evidence.owner_id
                })
            if hasattr(evidence, 'date_updated') and evidence.date_updated:
                details['audit_trail'].append({
                    'action': 'updated',
                    'date': evidence.date_updated.isoformat(),
                    'user_id': evidence.owner_id
                })
        except:
            pass  # Audit trail optional
        
        return details
    
    @staticmethod
    def duplicate_evidence(evidence_id, new_name, user_id):
        """
        Duplicate evidence with a new name
        
        Args:
            evidence_id: Original evidence ID
            new_name: New evidence name
            user_id: User ID performing the action
        
        Returns:
            Duplicated evidence object
        """
        from app.models import ProjectEvidence
        from app import db
        
        original = ProjectEvidence.query.get(evidence_id)
        if not original:
            raise ValueError(f"Original evidence not found: {evidence_id}")
        
        # Check if name already exists
        existing = ProjectEvidence.query.filter_by(
            name=new_name,
            tenant_id=original.tenant_id,
            project_id=original.project_id
        ).first()
        
        if existing:
            raise ValueError(f"Evidence with name '{new_name}' already exists")
        
        # Create duplicate
        duplicate = ProjectEvidence(
            name=new_name,
            description=original.description,
            content=original.content,
            group=original.group,
            collected_on=original.collected_on,
            source_type=original.source_type,
            evidence_type=original.evidence_type,
            compliance_standard=original.compliance_standard,
            severity=original.severity,
            needs_review=original.needs_review,
            extraction_metadata=original.extraction_metadata,
            source_document=original.source_document,
            source_reference=original.source_reference,
            is_global=original.is_global,
            owner_id=user_id,
            tenant_id=original.tenant_id,
            project_id=original.project_id
        )
        
        db.session.add(duplicate)
        db.session.commit()
        
        # Duplicate file if exists (this would need file copying logic)
        # For now, just copy the file reference
        if original.file_name:
            duplicate.file_name = original.file_name
            duplicate.file_provider = original.file_provider
            db.session.commit()
        
        # Log the action
        try:
            if hasattr(original.tenant, 'add_log'):
                original.tenant.add_log(
                    message=f"Duplicated evidence '{original.name}' to '{new_name}'",
                    namespace="evidence",
                    action="duplicate",
                    user_id=user_id,
                    meta={
                        'original_id': evidence_id,
                        'duplicate_id': duplicate.id
                    }
                )
        except:
            pass
        
        return duplicate
    
    @staticmethod
    def export_evidence(evidence_ids, export_format='json'):
        """
        Export evidence items
        
        Args:
            evidence_ids: List of evidence IDs
            export_format: Export format ('json' or 'csv')
        
        Returns:
            Export data
        """
        from app.models import ProjectEvidence
        
        evidence_items = []
        for evidence_id in evidence_ids:
            evidence = ProjectEvidence.query.get(evidence_id)
            if evidence:
                evidence_items.append(evidence.to_global_dict())
        
        if export_format == 'json':
            return json.dumps(evidence_items, indent=2, default=str)
        elif export_format == 'csv':
            # Basic CSV implementation
            import csv
            import io
            
            if not evidence_items:
                return ''
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=evidence_items[0].keys())
            writer.writeheader()
            writer.writerows(evidence_items)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    @staticmethod
    def get_evidence_by_source_type(tenant_id, source_type):
        """
        Get evidence by source type
        
        Args:
            tenant_id: Tenant ID
            source_type: Source type to filter by
        
        Returns:
            List of evidence items
        """
        from app.models import ProjectEvidence
        
        evidence_items = ProjectEvidence.query.filter(
            ProjectEvidence.tenant_id == tenant_id,
            ProjectEvidence.source_type == source_type
        ).order_by(
            ProjectEvidence.date_added.desc()
        ).all()
        
        return [evidence.to_global_dict() for evidence in evidence_items]
    
    @staticmethod
    def get_evidence_timeline(tenant_id, days=30):
        """
        Get evidence added in the last N days
        
        Args:
            tenant_id: Tenant ID
            days: Number of days to look back
        
        Returns:
            Timeline data
        """
        from app.models import ProjectEvidence
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        evidence_items = ProjectEvidence.query.filter(
            ProjectEvidence.tenant_id == tenant_id,
            ProjectEvidence.date_added >= cutoff_date
        ).order_by(
            ProjectEvidence.date_added.desc()
        ).all()
        
        # Group by day
        timeline = {}
        for evidence in evidence_items:
            date_str = evidence.date_added.date().isoformat() if evidence.date_added else 'unknown'
            if date_str not in timeline:
                timeline[date_str] = {
                    'date': date_str,
                    'count': 0,
                    'evidence': []
                }
            
            timeline[date_str]['count'] += 1
            timeline[date_str]['evidence'].append({
                'id': evidence.id,
                'name': evidence.name,
                'source_type': evidence.source_type
            })
        
        # Convert to list and sort by date
        timeline_list = sorted(
            timeline.values(),
            key=lambda x: x['date'],
            reverse=True
        )
        
        return timeline_list
    
    @staticmethod
    def validate_evidence_data(data):
        """
        Validate evidence data before creation/update
        
        Args:
            data: Evidence data dictionary
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check required fields
        if not data.get('name'):
            errors.append("Evidence name is required")
        
        # Validate source_type
        valid_source_types = ['manual', 'ai_ocr', 'upload', 'scan']
        source_type = data.get('source_type', 'manual')
        if source_type not in valid_source_types:
            errors.append(f"Invalid source_type. Must be one of: {', '.join(valid_source_types)}")
        
        # Validate severity if provided
        valid_severities = ['low', 'medium', 'high', 'critical', None]
        severity = data.get('severity')
        if severity not in valid_severities:
            errors.append(f"Invalid severity. Must be one of: {', '.join([s for s in valid_severities if s])}")
        
        # Validate needs_review
        needs_review = data.get('needs_review')
        if needs_review is not None and not isinstance(needs_review, bool):
            try:
                # Try to convert string to boolean
                if isinstance(needs_review, str):
                    data['needs_review'] = needs_review.lower() in ['true', '1', 'yes', 't']
                else:
                    errors.append("needs_review must be a boolean")
            except:
                errors.append("needs_review must be a boolean")
        
        return len(errors) == 0, errors

# Helper functions for the global evidence repository
class EvidenceUtils:
    """Utility functions for evidence management"""
    
    @staticmethod
    def generate_evidence_id():
        """Generate a unique evidence ID"""
        import shortuuid
        return str(shortuuid.ShortUUID().random(length=8)).lower()
    
    @staticmethod
    def format_content_preview(content, max_length=200):
        """Create a preview of evidence content"""
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        return content[:max_length] + '...'
    
    @staticmethod
    def get_evidence_type_display(evidence_type):
        """Get display name for evidence type"""
        display_map = {
            'policy': 'Policy Document',
            'screenshot': 'Screenshot',
            'log': 'Log File',
            'config': 'Configuration',
            'procedure': 'Procedure',
            'assessment': 'Assessment Result',
            'test': 'Test Result',
            'certificate': 'Certificate',
            'report': 'Report',
            'other': 'Other'
        }
        return display_map.get(evidence_type, evidence_type or 'Unknown')
    
    @staticmethod
    def get_severity_color(severity):
        """Get CSS color class for severity"""
        color_map = {
            'low': 'badge-success',
            'medium': 'badge-warning',
            'high': 'badge-error',
            'critical': 'badge-error'
        }
        return color_map.get(severity, 'badge-ghost')
    
    @staticmethod
    def get_source_type_icon(source_type):
        """Get icon class for source type"""
        icon_map = {
            'manual': 'ti ti-file-text',
            'ai_ocr': 'ti ti-robot',
            'upload': 'ti ti-upload',
            'scan': 'ti ti-scan'
        }
        return icon_map.get(source_type, 'ti ti-file')