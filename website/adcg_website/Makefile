DOCS= index install tutorial

DOCSH=$(addsuffix .html,$(DOCS))

.PHONY : docs
docs : $(DOCSH)

.PHONY : papers
papers :
	(cd papers; jemdoc -c *.jemdoc)

%.html : %.jemdoc
	jemdoc -c jemdoc.conf -o $@ $<

.PHONY : clean
clean :
	-rm -f *.html

	.PHONY : update
update : $(DOCSH)
	@echo 'Copying to live server...'
	@rsync -avz --no-p --exclude=Makefile --copy-unsafe-links --exclude=MENU --exclude=*.jemdoc --delete --delete-excluded * arwen.berkeley.edu:/mirror/data/pub/users/nickboyd/adcg/
	@echo ' done.'
