from sqlalchemy import inspect

from backend.app.db.database import engine, Base
from backend.app.models import (
    User,
    ModelMetadata,
    Prediction,
    PredictionExplanation
)


# Create an inspector for the connected MySQL database
inspector = inspect(engine)


# Get all table names registered in our ORM
table_names = sorted(Base.metadata.tables.keys())


# Inspect each table
for table_name in table_names:

    print(f"\n========== {table_name.upper()} ==========")

    # Get column information from MySQL
    mysql_columns = inspector.get_columns(table_name)

    # Get column definitions from our Python ORM
    orm_table = Base.metadata.tables[table_name]

    # Create a dictionary for easy column lookup
    mysql_column_map = {
        column["name"]: column
        for column in mysql_columns
    }

    # Compare ORM and MySQL column types
    for orm_column in orm_table.columns:

        mysql_column = mysql_column_map.get(orm_column.name)

        orm_type = orm_column.type.compile(dialect=engine.dialect)

        mysql_type = mysql_column["type"]

        mysql_unsigned = getattr(mysql_type, "unsigned", None)

        mysql_enum_values = getattr(mysql_type, "enums", None)

        print(
             f"{orm_column.name}: "
             f"ORM = {orm_type} | "
             f"MySQL = {mysql_type} | "
             f"UNSIGNED = {mysql_unsigned} | "
             f"ENUM VALUES = {mysql_enum_values}"
        )

    # Display primary key information
    primary_key = inspector.get_pk_constraint(table_name)

    print("PRIMARY KEY:", primary_key["constrained_columns"])

    # Display foreign key information
    foreign_keys = inspector.get_foreign_keys(table_name)

    if foreign_keys:
        for foreign_key in foreign_keys:
            print(
                "FOREIGN KEY:",
                foreign_key["constrained_columns"],
                "->",
                foreign_key["referred_table"],
                foreign_key["referred_columns"],
                "| ON DELETE:",
                foreign_key.get("options", {}).get("ondelete")
            )
    else:
        print("FOREIGN KEYS: None")