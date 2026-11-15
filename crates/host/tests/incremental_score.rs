use flatppl_host::{Context, EmitOptions};

#[test]
fn record_field_edits_preserve_sibling_terms_and_validate_prefixes_atomically() {
    use flatppl_host::InputPath;
    let path = |fields: &[&str]| InputPath {
        input: "point".into(),
        fields: fields.iter().map(|s| (*s).into()).collect(),
    };
    let mut context = Context::default();
    let module=context.parse("point=elementof(cartprod(a=reals,b=reals,nested=cartprod(a=reals,ab=reals),v=cartpow(reals,2)))\ninputs=point\noutputs=point.a*point.a+point.b*point.b+sum(point.v)+point.nested.a+point.nested.ab",None,None).unwrap();
    let plan = context.prepare_incremental_score(&module, &[]).unwrap();
    assert_eq!(plan.affected_paths(&[path(&["b"])]).unwrap(), [1]);
    assert_eq!(plan.affected_paths(&[path(&["nested"])]).unwrap(), [3, 4]);
    assert_eq!(plan.affected_paths(&[path(&["nested", "a"])]).unwrap(), [3]);
    assert_eq!(plan.affected_paths(&[path(&["v"])]).unwrap(), [2]);
    assert_eq!(plan.affected_paths(&[path(&[])]).unwrap(), [0, 1, 2, 3, 4]);
    let mut revisions = plan.revisions();
    let b = revisions.ticket(1).unwrap();
    revisions.invalidate_paths(&[path(&["a"])]).unwrap();
    assert!(revisions.accept(b));
    let a = revisions.ticket(0).unwrap();
    assert!(
        revisions
            .invalidate_paths(&[path(&["a"]), path(&["missing"])])
            .is_err()
    );
    assert!(revisions.accept(a));
    for term in plan.terms() {
        term.query.compile(&EmitOptions::default()).unwrap();
    }
}

fn prepared() -> (Context, flatppl_host::IncrementalScoreQueries) {
    let mut context = Context::default();
    let module = context.parse("a=elementof(reals)\nd=external(reals)\nb=elementof(reals)\nu=elementof(reals)\nf=logdensityof(Normal(a,1.0),d)\ng=logdensityof(Normal(b,2.0),d)\ninputs=(a,d,b,u)\noutputs=f+g-d*d/2.0", None, None).unwrap();
    let before = module.compile(&EmitOptions::default()).unwrap().stablehlo;
    let queries = context
        .prepare_incremental_score(&module, &["d".into()])
        .unwrap();
    assert_eq!(
        module.compile(&EmitOptions::default()).unwrap().stablehlo,
        before
    );
    (context, queries)
}

#[test]
fn promoted_alias_chains_keep_the_declared_dependency_name() {
    for source in [
        "a=elementof(reals)\nalias=a\ninputs=alias\noutputs=a+1.0",
        "a=2.0\nalias=a\ninputs=alias\noutputs=a+1.0",
        "a=load_data(\"missing-incremental-alias.json\",reals)\nalias=a\ninputs=alias\noutputs=a+1.0",
        "a=elementof(reals)\nalias=a\nouter=alias\ninputs=outer\noutputs=a+alias",
    ] {
        let mut context = Context::default();
        let module = context.parse(source, None, None).unwrap();
        let declared = if source.contains("outer=") {
            "outer"
        } else {
            "alias"
        };
        let (_, analysis) = context.analyze_model(&module, &[declared.into()]).unwrap();
        assert!(analysis.terms.iter().any(|term| term.data == [declared]));
        let queries = context
            .prepare_incremental_score(&module, &[declared.into()])
            .unwrap();
        assert_eq!(queries.affected_terms(&[declared.into()]).unwrap(), [0]);
        assert_eq!(queries.terms()[0].dependencies, [declared]);
        queries.terms()[0]
            .query
            .compile(&EmitOptions::default())
            .unwrap();
        queries.combine().compile(&EmitOptions::default()).unwrap();
    }
}

#[test]
fn term_queries_keep_dependency_order_and_an_ordinary_combiner() {
    let (_, queries) = prepared();
    assert_eq!(queries.inputs(), ["a", "d", "b", "u"]);
    assert_eq!(
        queries
            .terms()
            .iter()
            .map(|term| term.dependencies.clone())
            .collect::<Vec<_>>(),
        [vec!["a", "d"], vec!["d", "b"], vec!["d"]]
    );
    for term in queries.terms() {
        let export = term.query.compile(&EmitOptions::default()).unwrap();
        assert_eq!(
            export
                .inputs
                .iter()
                .map(|field| field.name.as_str())
                .collect::<Vec<_>>(),
            term.dependencies
        );
    }
    let combine = queries.combine().compile(&EmitOptions::default()).unwrap();
    assert_eq!(
        combine
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        queries
            .terms()
            .iter()
            .map(|term| term.cache_input.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(queries.affected_terms(&["a".into()]).unwrap(), [0]);
    assert_eq!(queries.affected_terms(&["d".into()]).unwrap(), [0, 1, 2]);
    assert!(queries.affected_terms(&["u".into()]).unwrap().is_empty());
}

#[test]
fn explicit_revisions_reject_stale_results_and_keep_unknown_notifications_atomic() {
    let (_, queries) = prepared();
    let mut revisions = queries.revisions();
    let old = revisions.ticket(0).unwrap();
    assert_eq!(
        revisions.invalidate(&["a".into(), "a".into()]).unwrap(),
        [0]
    );
    assert!(!revisions.accept(old));
    assert!(!revisions.is_current(0).unwrap());
    let current = revisions.ticket(0).unwrap();
    assert!(revisions.accept(current));
    assert!(revisions.is_current(0).unwrap());
    let unaffected = revisions.ticket(1).unwrap();
    revisions.invalidate(&["a".into()]).unwrap();
    assert!(revisions.accept(unaffected));
    let proposed = revisions.ticket(0).unwrap();
    // Restoring a rejected proposal is another explicit input change.
    revisions.invalidate(&["a".into()]).unwrap();
    assert!(!revisions.accept(proposed));
    let unchanged = revisions.ticket(0).unwrap();
    assert!(revisions.invalidate(&["a".into(), "typo".into()]).is_err());
    assert!(revisions.accept(unchanged));
    let another_cache = queries.revisions();
    assert!(!revisions.accept(another_cache.ticket(0).unwrap()));
    let shared_data = revisions.ticket(1).unwrap();
    assert_eq!(revisions.invalidate(&["d".into()]).unwrap(), [0, 1, 2]);
    assert!(!revisions.accept(shared_data));
}

#[test]
fn repeated_signed_terms_and_integer_scores_compile_without_changing_the_original() {
    for source in [
        "a=external(reals)\nb=external(reals)\nc=external(reals)\ninputs=(a,b,c)\noutputs=(a+b)-(c+a)",
        "a=external(integers)\nb=external(integers)\nf=2*a\ninputs=(a,b)\noutputs=f-(b-f)",
        "a=external(reals)\ninputs=a\noutputs=logdensityof(Exponential(1.0),a)+logdensityof(Exponential(1.0),-1.0)",
    ] {
        let mut context = Context::default();
        let module = context.parse(source, None, None).unwrap();
        let queries = context.prepare_incremental_score(&module, &[]).unwrap();
        for term in queries.terms() {
            term.query.compile(&EmitOptions::default()).unwrap();
        }
        queries.combine().compile(&EmitOptions::default()).unwrap();
        assert_eq!(
            queries.terms().len(),
            if source.contains("inputs=(a,b,c)") {
                3
            } else {
                2
            }
        );
    }
}
