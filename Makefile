SRCPATH=./src/main/java/uk/ac/soton/
BINPATH=./bin
BASEPKG=uk.ac.soton.
## Comile all .java files in src source to bin/.
all: .classpath r .sources
	javac -d ${BINPATH} -cp "`cat .classpath`" @.sources

## Create a runner script.
r:
	echo '#!/bin/sh\njava -cp "${BINPATH}:`cat .classpath`" ${BASEPKG}$$@\n' > r
	chmod +x r

## Download all dependencies
fetch: pom.xml
	mvn dependency:resolve -q

## Delete compiled files (.class) and source files index.
clean:
	-rm -r bin/*
	-rm .sources

## Delete compiled files (.class), .classpath and the runner script.
refresh: clean
	-rm .classpath
	-rm r

## Create source files index
.sources: ${SRCPATH}
	find ${SRCPATH} -name *.java > .sources

## Generate the classpath
.classpath: pom.xml
	mvn dependency:build-classpath -Dmdep.pathSeparator=":" -Dmdep.fileSeparator="/" -Dmdep.outputFile=.classpath -q
	touch .classpath
