setup:
	pip install pipenv
	pipenv sync
	pipenv shell

dev:
	pip install pipenv
	pipenv sync --dev
	pipenv shell

rm:
	pipenv --rm

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

ifneq (,$(findstring release-,$(BRANCH)))
VERSION := $(subst release-,,$(BRANCH))
else ifneq (,$(findstring hotfix-,$(BRANCH)))
VERSION := $(subst hotfix-,,$(BRANCH))
endif

bump: $(shell find . -name "*.py")
	sed -i '' 's/__version__ = .*/__version__ = '\'$(VERSION)\''/' $^
	autopep8 -i -a -a $^
	pdoc -o ./docs --docformat numpy aml
	pipenv lock
	git add .
	git commit -m "Bump version number to $(VERSION)"
	git checkout master
	git merge $(BRANCH)
	git tag $(VERSION)
	git checkout develop
	git merge $(BRANCH)
	git branch -d $(BRANCH)