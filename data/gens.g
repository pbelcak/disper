targetFile := "gens20.csv";

rolling_id := 0;
for order in [1..20] do
	for id in [1..NrSmallGroups(order)] do
		rolling_id := rolling_id + 1;

		G := SmallGroup(order, id);
		AppendTo(targetFile, order);
		AppendTo(targetFile, ";");
		AppendTo(targetFile, id);
		AppendTo(targetFile, ";");
		AppendTo(targetFile, rolling_id);
		AppendTo(targetFile, ";");
		AppendTo(targetFile, StructureDescription(G));
		AppendTo(targetFile, ";");
		AppendTo(targetFile, MinimalGeneratingSet(G));
		AppendTo(targetFile, ";");
		AppendTo(targetFile, MultiplicationTable(G));
		AppendTo(targetFile, "\n");
	od;
od;