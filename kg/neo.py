from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable

class App:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def create_sector(self, ticker1, ticker2, category, relation):
        with self.driver.session(database="neo4j") as session:
            # Write transactions allow the driver to handle retries and transient errors
            result = session.execute_write(
                self._create_and_return_sector, ticker1, ticker2, category, relation)
            for row in result:
                print("Created node between: {p1}, {p2}".format(p1=row['p1'], p2=row['p2']))

    @staticmethod
    def _create_and_return_sector(tx, ticker1, ticker2, category, relation):
        # To learn more about the Cypher syntax, see https://neo4j.com/docs/cypher-manual/current/
        # The Reference Card is also a good resource for keywords https://neo4j.com/docs/cypher-refcard/current/
        
        query = (
            
            "MATCH (a:Asset) WHERE a.ticker = $ticker1 "
            "RETURN a"
        )
        result = tx.run(query, ticker1=ticker1, ticker2=ticker2)

        if len(list(result)) == 0:
            print("creating asset 1")
            query = (
            "CREATE (p1:Asset { ticker: $ticker1 }) "
            )
            result = tx.run(query, ticker1=ticker1, ticker2=ticker2)
        else:
            return [{"p1": "place", "p2": "[place]"}]

        query = (
            "MATCH (c:Classification) WHERE c.category = $category AND c.id = $ticker2 "
            "RETURN c"
        )
        result = tx.run(query, ticker1=ticker1, ticker2=ticker2, category=category)
        
        if len(list(result)) == 0:
            query = (
                "CREATE (c1:Classification { category: $category, id: $ticker2 }) "
            )
            result = tx.run(query, ticker1=ticker1, ticker2=ticker2, category=category)

        query = (
            "MATCH (a:Asset), (c:Classification) WHERE a.ticker = $ticker1 AND c.id = $ticker2 "
            "CREATE (a)-[:"+relation+"]->(c) "
            "RETURN a, c"
        )
        result = tx.run(query, ticker1=ticker1, ticker2=ticker2, category=category)
        try:
            return [{"p1": row["a"]["ticker"], "p2": row["c"]["category"]}
                    for row in result]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def find_person(self, person_name):
        with self.driver.session(database="neo4j") as session:
            result = session.execute_read(self._find_and_return_person, person_name)
            for row in result:
                print("Found person: {row}".format(row=row))

    @staticmethod
    def _find_and_return_person(tx, person_name):
        query = (
            "MATCH (p:Person) "
            "WHERE p.name = $person_name "
            "RETURN p.name AS name"
        )
        result = tx.run(query, person_name=person_name)
        return [row["name"] for row in result]

def add_sector_nodes(app, sector_graph):

    for lines in sector_graph[1:]:
        lines = lines[:-1]
        tickers = lines.split("\t")[2:]
        category = lines.split("\t")[1]
        relation = lines.split("\t")[0]

        if len(relation) == 2:
            rel = "SECTOR"
        elif len(relation) == 4:
            rel = "INDUSTRY"
        elif len(relation) == 6:
            rel = "INDUSTRY_GROUP"
        else:
            rel = "SUB_INDUSTRY"

        for i in range(len(tickers)):
            app.create_sector(tickers[i], str(relation), category, rel)


if __name__ == "__main__":
    # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
    uri = "neo4j+s://8be78757.databases.neo4j.io"
    user = "neo4j"
    password = "TAGiGB_xYviLbxGRjsl8fnfo57iEkuYKTdD5UjD7f38"
    app = App(uri, user, password)

    #sector_graph = open("sector/sector_hypergraph_nasdaq100.txt", "r").readlines()
    #add_sector_nodes(app, sector_graph)

    #sector_graph = open("sector/sector_hypergraph_sp500.txt", "r").readlines()
    #add_sector_nodes(app, sector_graph)
    app.close()