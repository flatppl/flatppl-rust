module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<5.0> : tensor<f32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.compare LT, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.add %0, %3 : tensor<f32>
    %6 = stablehlo.select %4, %5, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %8 = stablehlo.subtract %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<9.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.sqrt %10 : tensor<f32>
    %12 = stablehlo.divide %3, %11 : tensor<f32>
    %13 = stablehlo.constant dense<128> : tensor<1xi64>
    %14 = stablehlo.rng %2, %3, %13, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %15 = stablehlo.constant dense<128> : tensor<1xi64>
    %16 = stablehlo.rng %2, %3, %15, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %17 = stablehlo.constant dense<0> : tensor<i32>
    %18 = stablehlo.constant dense<false> : tensor<i1>
    %19 = stablehlo.constant dense<0.0> : tensor<f32>
    %23:3 = stablehlo.while(%20 = %17, %21 = %18, %22 = %19) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %24 = stablehlo.constant dense<128> : tensor<i32>
      %25 = stablehlo.compare LT, %20, %24, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %26 = stablehlo.not %21 : tensor<i1>
      %27 = stablehlo.and %26, %25 : tensor<i1>
      stablehlo.return %27 : tensor<i1>
    } do {
      %28 = stablehlo.dynamic_slice %14, %20, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %29 = stablehlo.reshape %28 : (tensor<1xf32>) -> tensor<f32>
      %30 = stablehlo.dynamic_slice %16, %20, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %31 = stablehlo.reshape %30 : (tensor<1xf32>) -> tensor<f32>
      %32 = stablehlo.multiply %12, %29 : tensor<f32>
      %33 = stablehlo.add %3, %32 : tensor<f32>
      %34 = stablehlo.multiply %33, %33 : tensor<f32>
      %35 = stablehlo.multiply %34, %33 : tensor<f32>
      %36 = stablehlo.multiply %8, %35 : tensor<f32>
      %37 = stablehlo.constant dense<0.5> : tensor<f32>
      %38 = stablehlo.multiply %29, %29 : tensor<f32>
      %39 = stablehlo.multiply %37, %38 : tensor<f32>
      %40 = stablehlo.multiply %8, %35 : tensor<f32>
      %41 = stablehlo.negate %40 : tensor<f32>
      %42 = stablehlo.log %35 : tensor<f32>
      %43 = stablehlo.multiply %8, %42 : tensor<f32>
      %44 = stablehlo.add %39, %8 : tensor<f32>
      %45 = stablehlo.add %44, %41 : tensor<f32>
      %46 = stablehlo.add %45, %43 : tensor<f32>
      %47 = stablehlo.log %31 : tensor<f32>
      %48 = stablehlo.compare LT, %47, %46 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %49 = stablehlo.compare GT, %35, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %50 = stablehlo.and %48, %49 : tensor<i1>
      %51 = stablehlo.constant dense<1> : tensor<i32>
      %52 = stablehlo.add %20, %51 : tensor<i32>
      stablehlo.return %52, %50, %36 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %53 = stablehlo.constant dense<> : tensor<0xi64>
    %54 = stablehlo.rng %2, %3, %53, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %55 = stablehlo.divide %3, %0 : tensor<f32>
    %56 = stablehlo.power %54, %55 : tensor<f32>
    %57 = stablehlo.select %4, %56, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %58 = stablehlo.multiply %23#2, %57 : tensor<f32>
    %59 = stablehlo.divide %58, %1 : tensor<f32>
    %60 = stablehlo.constant dense<0.0> : tensor<f32>
    %61 = stablehlo.constant dense<1.0> : tensor<f32>
    %62 = stablehlo.constant dense<> : tensor<0xi64>
    %63 = stablehlo.rng %60, %61, %62, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %64 = stablehlo.negate %59 : tensor<f32>
    %65 = stablehlo.exponential %64 : tensor<f32>
    %66 = stablehlo.constant dense<0.0> : tensor<f32>
    %67 = stablehlo.constant dense<false> : tensor<i1>
    %68 = stablehlo.constant dense<0.0> : tensor<f32>
    %74:5 = stablehlo.while(%69 = %66, %70 = %65, %71 = %65, %72 = %67, %73 = %68) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %75 = stablehlo.constant dense<256.0> : tensor<f32>
      %76 = stablehlo.compare LT, %69, %75 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %77 = stablehlo.not %72 : tensor<i1>
      %78 = stablehlo.and %77, %76 : tensor<i1>
      stablehlo.return %78 : tensor<i1>
    } do {
      %79 = stablehlo.compare LE, %63, %70 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %80 = stablehlo.constant dense<1.0> : tensor<f32>
      %81 = stablehlo.add %69, %80 : tensor<f32>
      %82 = stablehlo.divide %59, %81 : tensor<f32>
      %83 = stablehlo.multiply %71, %82 : tensor<f32>
      %84 = stablehlo.add %70, %83 : tensor<f32>
      stablehlo.return %81, %84, %83, %79, %69 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %74#4 : tensor<f32>
  }
}
