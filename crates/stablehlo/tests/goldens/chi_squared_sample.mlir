module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.multiply %1, %0 : tensor<f32>
    %3 = stablehlo.constant dense<0.5> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.compare LT, %2, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.add %2, %5 : tensor<f32>
    %8 = stablehlo.select %6, %7, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %10 = stablehlo.subtract %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<9.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.sqrt %12 : tensor<f32>
    %14 = stablehlo.divide %5, %13 : tensor<f32>
    %15 = stablehlo.constant dense<128> : tensor<1xi64>
    %16 = stablehlo.rng %4, %5, %15, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %17 = stablehlo.constant dense<128> : tensor<1xi64>
    %18 = stablehlo.rng %4, %5, %17, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %19 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.constant dense<false> : tensor<i1>
    %21 = stablehlo.constant dense<0.0> : tensor<f32>
    %25:3 = stablehlo.while(%22 = %19, %23 = %20, %24 = %21) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %26 = stablehlo.constant dense<128> : tensor<i32>
      %27 = stablehlo.compare LT, %22, %26, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %28 = stablehlo.not %23 : tensor<i1>
      %29 = stablehlo.and %28, %27 : tensor<i1>
      stablehlo.return %29 : tensor<i1>
    } do {
      %30 = stablehlo.dynamic_slice %16, %22, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %31 = stablehlo.reshape %30 : (tensor<1xf32>) -> tensor<f32>
      %32 = stablehlo.dynamic_slice %18, %22, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %33 = stablehlo.reshape %32 : (tensor<1xf32>) -> tensor<f32>
      %34 = stablehlo.multiply %14, %31 : tensor<f32>
      %35 = stablehlo.add %5, %34 : tensor<f32>
      %36 = stablehlo.multiply %35, %35 : tensor<f32>
      %37 = stablehlo.multiply %36, %35 : tensor<f32>
      %38 = stablehlo.multiply %10, %37 : tensor<f32>
      %39 = stablehlo.constant dense<0.5> : tensor<f32>
      %40 = stablehlo.multiply %31, %31 : tensor<f32>
      %41 = stablehlo.multiply %39, %40 : tensor<f32>
      %42 = stablehlo.multiply %10, %37 : tensor<f32>
      %43 = stablehlo.negate %42 : tensor<f32>
      %44 = stablehlo.log %37 : tensor<f32>
      %45 = stablehlo.multiply %10, %44 : tensor<f32>
      %46 = stablehlo.add %41, %10 : tensor<f32>
      %47 = stablehlo.add %46, %43 : tensor<f32>
      %48 = stablehlo.add %47, %45 : tensor<f32>
      %49 = stablehlo.log %33 : tensor<f32>
      %50 = stablehlo.compare LT, %49, %48 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %51 = stablehlo.compare GT, %37, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %52 = stablehlo.and %50, %51 : tensor<i1>
      %53 = stablehlo.constant dense<1> : tensor<i32>
      %54 = stablehlo.add %22, %53 : tensor<i32>
      stablehlo.return %54, %52, %38 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %55 = stablehlo.constant dense<> : tensor<0xi64>
    %56 = stablehlo.rng %4, %5, %55, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %57 = stablehlo.divide %5, %2 : tensor<f32>
    %58 = stablehlo.power %56, %57 : tensor<f32>
    %59 = stablehlo.select %6, %58, %5 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %60 = stablehlo.multiply %25#2, %59 : tensor<f32>
    %61 = stablehlo.divide %60, %3 : tensor<f32>
    return %61 : tensor<f32>
  }
}
