module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.constant dense<5.0> : tensor<f32>
    %2 = stablehlo.divide %1, %0 : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %1, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %1, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14 = stablehlo.constant dense<128> : tensor<1xi64>
    %15 = stablehlo.rng %3, %4, %14, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %16 = stablehlo.constant dense<128> : tensor<1xi64>
    %17 = stablehlo.rng %3, %4, %16, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.constant dense<false> : tensor<i1>
    %20 = stablehlo.constant dense<0.0> : tensor<f32>
    %24:3 = stablehlo.while(%21 = %18, %22 = %19, %23 = %20) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %25 = stablehlo.constant dense<128> : tensor<i32>
      %26 = stablehlo.compare LT, %21, %25, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %27 = stablehlo.not %22 : tensor<i1>
      %28 = stablehlo.and %27, %26 : tensor<i1>
      stablehlo.return %28 : tensor<i1>
    } do {
      %29 = stablehlo.dynamic_slice %15, %21, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
      %31 = stablehlo.dynamic_slice %17, %21, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
      %33 = stablehlo.multiply %13, %30 : tensor<f32>
      %34 = stablehlo.add %4, %33 : tensor<f32>
      %35 = stablehlo.multiply %34, %34 : tensor<f32>
      %36 = stablehlo.multiply %35, %34 : tensor<f32>
      %37 = stablehlo.multiply %9, %36 : tensor<f32>
      %38 = stablehlo.constant dense<0.5> : tensor<f32>
      %39 = stablehlo.multiply %30, %30 : tensor<f32>
      %40 = stablehlo.multiply %38, %39 : tensor<f32>
      %41 = stablehlo.multiply %9, %36 : tensor<f32>
      %42 = stablehlo.negate %41 : tensor<f32>
      %43 = stablehlo.log %36 : tensor<f32>
      %44 = stablehlo.multiply %9, %43 : tensor<f32>
      %45 = stablehlo.add %40, %9 : tensor<f32>
      %46 = stablehlo.add %45, %42 : tensor<f32>
      %47 = stablehlo.add %46, %44 : tensor<f32>
      %48 = stablehlo.log %32 : tensor<f32>
      %49 = stablehlo.compare LT, %48, %47 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %50 = stablehlo.compare GT, %36, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %51 = stablehlo.and %49, %50 : tensor<i1>
      %52 = stablehlo.constant dense<1> : tensor<i32>
      %53 = stablehlo.add %21, %52 : tensor<i32>
      stablehlo.return %53, %51, %37 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %54 = stablehlo.constant dense<> : tensor<0xi64>
    %55 = stablehlo.rng %3, %4, %54, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %56 = stablehlo.divide %4, %1 : tensor<f32>
    %57 = stablehlo.power %55, %56 : tensor<f32>
    %58 = stablehlo.select %5, %57, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %59 = stablehlo.multiply %24#2, %58 : tensor<f32>
    %60 = stablehlo.divide %59, %2 : tensor<f32>
    %61 = stablehlo.constant dense<0.0> : tensor<f32>
    %62 = stablehlo.constant dense<1.0> : tensor<f32>
    %63 = stablehlo.constant dense<> : tensor<0xi64>
    %64 = stablehlo.rng %61, %62, %63, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %65 = stablehlo.negate %60 : tensor<f32>
    %66 = stablehlo.exponential %65 : tensor<f32>
    %67 = stablehlo.constant dense<0.0> : tensor<f32>
    %68 = stablehlo.constant dense<false> : tensor<i1>
    %69 = stablehlo.constant dense<0.0> : tensor<f32>
    %75:5 = stablehlo.while(%70 = %67, %71 = %66, %72 = %66, %73 = %68, %74 = %69) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %76 = stablehlo.constant dense<256.0> : tensor<f32>
      %77 = stablehlo.compare LT, %70, %76 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %78 = stablehlo.not %73 : tensor<i1>
      %79 = stablehlo.and %78, %77 : tensor<i1>
      stablehlo.return %79 : tensor<i1>
    } do {
      %80 = stablehlo.compare LE, %64, %71 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %81 = stablehlo.constant dense<1.0> : tensor<f32>
      %82 = stablehlo.add %70, %81 : tensor<f32>
      %83 = stablehlo.divide %60, %82 : tensor<f32>
      %84 = stablehlo.multiply %72, %83 : tensor<f32>
      %85 = stablehlo.add %71, %84 : tensor<f32>
      stablehlo.return %82, %85, %84, %80, %70 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %75#4 : tensor<f32>
  }
}
