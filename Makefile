.PHONY: report report-en navigation check models package package-with-models package-all

navigation:
	python3 -B tools/build_navigation.py

report:
	python3 tools/build_catalog.py
	bash reports/zh/build.sh

report-en:
	bash reports/en/build.sh

check:
	python3 -B tools/build_navigation.py --check
	python3 -B tools/test_navigation.py
	python3 tools/validate_repository.py

models:
	python3 tools/fetch_models.py --weights

package:
	python3 tools/package_release.py

package-with-models:
	python3 tools/package_release.py --include-weights

package-all:
	python3 tools/package_release.py --both
