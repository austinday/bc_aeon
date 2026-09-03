from pathlib import Path

from aeon.core.skills.manager import SkillsManager


def test_adaylabs_publishing_skill_is_in_shared_catalog() -> None:
    manager = SkillsManager()
    assert "publishing" in manager.list_categories()
    assert "adaylabs_mail_and_sites" in manager.get_skills_in_category("publishing")
    content = manager.get_skill_content("publishing", "adaylabs_mail_and_sites")
    assert content is not None
    assert "## When to use" in content
    assert "/home/aday/website_hosting/ADAYLABS_PUBLISHING.md" in content
    assert "python manage.py create" in content
    assert "guidance, not authority" in content


def test_canonical_adaylabs_runbook_does_not_duplicate_the_credential() -> None:
    runbook = Path("/home/aday/website_hosting/ADAYLABS_PUBLISHING.md").read_text()
    assert "/home/aday/CloudfareCredentials.txt" in runbook
    assert "Never create a new copy" in " ".join(runbook.split())
    assert "name@adaylabs.com" in runbook
    assert "label.adaylabs.com" in runbook
