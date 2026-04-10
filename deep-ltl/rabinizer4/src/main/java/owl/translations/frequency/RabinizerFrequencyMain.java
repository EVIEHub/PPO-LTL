package owl.translations.frequency;

import java.util.EnumSet;
import org.apache.commons.cli.CommandLine;
import owl.factories.Factories;
import owl.ltl.LabelledFormula;
import owl.ltl.MOperator;
import owl.ltl.ROperator;
import owl.ltl.WOperator;
import owl.ltl.visitors.DefaultConverter;
import owl.ltl.visitors.UnabbreviateVisitor;
import owl.run.modules.InputReaders;
import owl.run.modules.OutputWriters;
import owl.run.modules.OwlModuleParser;
import owl.run.modules.Transformer;
import owl.run.modules.Transformers;
import owl.run.parser.PartialConfigurationParser;
import owl.run.parser.PartialModuleConfiguration;

public final class RabinizerFrequencyMain implements OwlModuleParser.TransformerParser {
  private static final UnabbreviateVisitor UNABBREVIATE_VISITOR =
    new UnabbreviateVisitor(ROperator.class, MOperator.class, WOperator.class);

  private RabinizerFrequencyMain() {}

  public static void main(String... args) {
    PartialConfigurationParser.run(args, PartialModuleConfiguration.builder("fltl2dgmra")
      .reader(InputReaders.LTL)
      .addTransformer(DefaultConverter.asTransformer(UNABBREVIATE_VISITOR))
      .addTransformer(new RabinizerFrequencyMain())
      .writer(OutputWriters.HOA)
      .build());
  }

  @Override
  public String getKey() {
    return "fltl2dgrma";
  }

  @Override
  public Transformer parse(CommandLine commandLine) {
    return environment ->
      Transformers.instanceFromFunction(LabelledFormula.class, (input) -> {
        Factories factories = environment.factorySupplier().getFactories(input);
        return new DtgrmaFactory(input.formula, factories, EnumSet.allOf(Optimisation.class))
          .constructAutomaton();
      });
  }
}
